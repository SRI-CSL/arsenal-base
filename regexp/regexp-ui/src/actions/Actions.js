import * as Actions from './ActionTypes';
// import { createRegexes } from '../regex';

export function showApiError(error){
    console.log("Error",error);
    return { type: Actions.SHOW_API_ERROR, error }
}

export function hideApiError(){
    return { type: Actions.HIDE_API_ERROR }
}

export function hideCST() {
    return { type: Actions.HIDE_CST }
}

export function showCST() {
    return { type: Actions.SHOW_CST }
}

export function hideEntities() {
    return { type: Actions.HIDE_ENTITIES }
}

export function showEntities() {
    return { type: Actions.SHOW_ENTITIES }
}

export function setNL(text) {
    return { type: Actions.SET_NL, text}
}

// export function setRegexes(text) {
//     return { type: Actions.SET_REGEXES, text}
// }

export function requestCSTResult(text) {
    return { type: Actions.REQUEST_CST_RESULT, text }
}
  
export function receiveCSTResult(result) {
    return { type: Actions.RECEIVE_CST_RESULT, result }
}

export function requestEntityResult(text) {
    return { type: Actions.REQUEST_ENTITY_RESULT, text }
}
  
export function receiveEntityResult(result) {
    return { type: Actions.RECEIVE_ENTITY_RESULT, result }
}

export function requestImprovedCSTResult(text) {
    return { type: Actions.REQUEST_IMPROVED_CST_RESULT, text }
}

export function receiveImprovedCSTResult(result) {
    return { type: Actions.RECEIVE_IMPROVED_CST_RESULT, result }
}

// Async "pseudo-actions" actions below
// These trigger two actions: A "request" action (immediately) and a "receive" action (some time later),
// or alternatively, an api error action if something went wrong.

// Apply a function to all primitive values in nested object/array.
// Creates and returns a new object, without modifying the original.
function mapDeep(val, func){
    if (val && Array.isArray(val)) {
        return val.map( (el) => mapDeep(el, func) )   // recurse in array
    } 
    else if (val && typeof val === 'object') {
        return Object.entries(val).reduce((newObj, [key, val]) => {
            return {...newObj, [key]: mapDeep(val, func)}    // recurse in nested object
        }, {})
    }
    else {
        return func(val)                   // apply func.
    }
}

function substituteEntities(cst, ep_result){
    let results = {
        sentences: cst.sentences.map( (s,idx) => {
            let substs = ep_result[idx].substitutions;
            let result = mapDeep(s, (val) => {
                if (Object.keys(substs).includes(val)){
                    return substs[val];
                }
                else {
                    return val;
                }
            });
            return result;
        })
    }
    return results;
}

// Creates proper JSON input for nlp2ir
function create_nl2cst_request(ep_result){
    return ep_result.map( (r) => {
        return {
            id: r.id,
            sentence: r['new-text']
        }
    });
}

// Creates proper JSON input for entity processor
function create_ep_request(text){
    let lines = text.split('\n');
    return lines.filter( (line) => line.trim() ).map( (line,index) => { 
        return {
            id: 'S' + (index+1),  // line numbers start from 1
            text: line
        };
    });
}

function create_reformulate_request(ep_sents, cst_sents){
    if (ep_sents.length !== cst_sents.length){
        throw Error("Different number of sentences from entity processor vs nl2cst!")
    }
    let result_arr = []
    for (let i=0; i<ep_sents.length; i++){
        let ep_obj = ep_sents[i]
        let cst_obj = cst_sents[i]
        if (ep_obj['id'] !== cst_obj['id']){
            throw Error("Inconsistent sentence IDs from entity processor vs nl2cst!")
        }
        result_arr.push({
            id: ep_obj['id'],
            'orig-text': ep_obj['orig-text'],
            cst: cst_obj['cst'],
            substitutions: ep_obj['substitutions']
        })
    }
    return result_arr
}

export function generateModel(text){
    return (dispatch,getState) => {
        dispatch(requestEntityResult(text));
        //fetch('/sal/nl2cst_dummy', { method: "POST", //headers: { "Content-Type": "application/json", "Accept": "application/json" },
        var ep_results = [];
        fetch('/entity/process_all', { method: "POST", headers: { "Content-Type": "application/json" },
                redirect: "follow", body: JSON.stringify({ 
                    sentences: create_ep_request(text)
                })
            })
            .then( (response) => {
                if (response.ok) return response.json();
                else return response.text().then(text => {throw Error(text)});
            })
            .then( (json) => {
                dispatch(receiveEntityResult(json));
                ep_results = json['sentences'];
                let req = create_nl2cst_request(ep_results)
                dispatch(requestCSTResult(req));
                return fetch('/nl2cst/generateir', { method: "POST", //headers: { "Content-Type": "application/json", "Accept": "application/json" },
                    redirect: "follow", body: JSON.stringify({
                            Username: "TestUser",
                            GroupName: "TestGroup",
                            msg: req
                        })    
                })
            })
            .then( (response) => {
                if (response.ok) return response.json();
                else return response.text().then(text => {throw Error(text)});
            })
            .then( (cst_json) => {
                console.log('Got CSTs',cst_json)
                dispatch(receiveCSTResult(cst_json));
                let cst_results = cst_json['sentences']
                dispatch(requestImprovedCSTResult(cst_results));
                return fetch('/reformulate/', { method: "POST",
                    redirect: "follow", body: JSON.stringify({
                        sentences: create_reformulate_request(ep_results, cst_results)
                    })
                })
            })
            .then( (response) => {
                if (response.ok) return response.json();
                else return response.text().then(text => {throw Error(text)});
            })
            .then( (json) => {
                console.log('Got reformulated CSTs',json)
                dispatch(receiveImprovedCSTResult(json));
            })
            // .then( (json) => {
            //     let realJson = substituteEntities(json, ep_results);
            //     dispatch(receiveCSTResult(realJson));
            //     let regexes = createRegexes(realJson);
            //     dispatch(setRegexes(regexes));
            // })
            .catch( (error) => dispatch(showApiError(error)) );
    }

}
