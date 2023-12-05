import * as Actions from './ActionTypes';

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

export function clearResults() {
    return { type: Actions.CLEAR_RESULTS }
}

export function setNL(text) {
    return { type: Actions.SET_NL, text}
}

export function setEntities(text) {
    return { type: Actions.SET_ENTITIES, text}
}

export function requestCSTResult(text) {
    return { type: Actions.REQUEST_CST_RESULT, text }
}
  
export function receiveCSTResult(result) {
    return { type: Actions.RECEIVE_CST_RESULT, result }
}

export function requestReformulationResult(text) {
    return { type: Actions.REQUEST_REFORMULATION_RESULT, text }
}
  
export function receiveReformulationResult(result) {
    return { type: Actions.RECEIVE_REFORMULATION_RESULT, result }
}

export function receiveImprovedCSTResult(result) {
    return { type: Actions.RECEIVE_IMPROVED_CST_RESULT, result }
}

export function requestEntityResult(text) {
    return { type: Actions.REQUEST_ENTITY_RESULT, text }
}
  
export function receiveEntityResult(result) {
    return { type: Actions.RECEIVE_ENTITY_RESULT, result }
}

export function setNoOpEP() {
    return { type: Actions.NOOP_ENTITY_PROCESSOR }
}

export function setRealEP() {
    return { type: Actions.REAL_ENTITY_PROCESSOR }
}

// Creates proper JSON input for entity processor
function create_ep_request(text){
    let lines = text.split('\n');
    return lines.filter( (line) => line.trim() ).map( (line,index) => { 
        //let text = removeParens(line)
        let text = line
        return {
            id: 'S' + (index+1),  // line numbers start from 1
            text: text
        };
    });
}

function create_cleaner_request(ep_result){
    return ep_result.map( s => s['new-text'])
}

function create_cleaner_result(cleaner_output, ep_results){
    let sentences = ep_results.map( (s,i) => {
        let new_sent = {...s}
        new_sent['new-text'] = cleaner_output[i]
        new_sent['precleaned-text'] = s['new-text']
        return new_sent
    })
    return sentences
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
            'orig-text': ep_obj['orig-text'],
            'ep-text': ep_obj['precleaned-text'],
            'cleaned-text': ep_obj['new-text'],
            id: ep_obj['id'],
            cst: cst_obj['cst'],
            substitutions: ep_obj['substitutions']
        })
    }
    return result_arr
}

function entityProcessorUrl(state){
    if (state.noopEntity){
        return "/noop_entity"
    }
    else {
        return "/entity"
    }
}

export function generateModel(text){
    return async (dispatch,getState) => {
        dispatch(clearResults())

        // 1. Call the Entity Processor
        dispatch(requestEntityResult(text));
        const epUrl = entityProcessorUrl(getState())
        let ep_response = await fetch(`${epUrl}/process_all`, { method: "POST", headers: { "Content-Type": "application/json" },
            redirect: "follow", body: JSON.stringify({ 
                sentences: create_ep_request(text)
            })
        })
        if (!ep_response.ok){
            let ep_text = await ep_response.text()
            throw Error(ep_text)
        }
        let ep_result = (await ep_response.json())['sentences']
        dispatch(receiveEntityResult(ep_result))

        // 2. Call the sentence cleaner (no UI updates)
        let cleaner_req = create_cleaner_request(ep_result)
        let cleaner_response = await fetch('/cleaner/clean', { method: "POST", //headers: { "Content-Type": "application/json", "Accept": "application/json" },
                redirect: "follow", body: JSON.stringify(cleaner_req)    
            })
        if (!cleaner_response.ok){
            let cleaner_text = await cleaner_response.text()
            throw Error(cleaner_text)
        }
        let cleaner_result = create_cleaner_result(await cleaner_response.json(), ep_result)

        // 3. Call nl2cst
        dispatch(requestCSTResult(cleaner_result));
        let nl2cst_response = await fetch('/nl2cst/generateir', { method: "POST", //headers: { "Content-Type": "application/json", "Accept": "application/json" },
            redirect: "follow", body: JSON.stringify({
                    sentences: cleaner_result
                })    
            })
        if (!nl2cst_response.ok){
            let nl2cst_text = await nl2cst_response.text()
            throw Error(nl2cst_text)
        }
        let nl2cst_result = (await nl2cst_response.json())['sentences']
        dispatch(receiveCSTResult(nl2cst_result))

        // 4. Call reformulate
        let reformulate_req = create_reformulate_request(cleaner_result, nl2cst_result)
        dispatch(requestReformulationResult(reformulate_req));
        let reformulate_response = await fetch('/reformulate/', { method: "POST",
            redirect: "follow", body: JSON.stringify({
                sentences: reformulate_req
            })
        })
        if (!reformulate_response.ok){
            let reformulate_text = await reformulate_response.text()
            throw Error(reformulate_text)
        }
        let reformulate_result = await reformulate_response.json()
        dispatch(receiveReformulationResult(reformulate_result))
        dispatch(receiveImprovedCSTResult(reformulate_result))
    }
}

export function regenerateModel(ep_string){
    return (dispatch,getState) => {
        var ep_json = JSON.parse(ep_string)
        dispatch(receiveEntityResult(ep_json));
        var ep_results = ep_json['sentences'];
        let req = create_nl2cst_request(ep_results)
        dispatch(requestCSTResult(req));
        fetch('/nl2cst/generateir', { method: "POST", //headers: { "Content-Type": "application/json", "Accept": "application/json" },
                                      redirect: "follow", body: JSON.stringify({
                                          Username: "TestUser",
                                          GroupName: "TestGroup",
                                          msg: req
                                      })    
                                    })
            .then( (cst_response) => {
                if (cst_response.ok) return cst_response.json();
                else return cst_response.text().then(text => {throw Error(text)});
            })
            .then( (cst_json) => {
                dispatch(receiveCSTResult(cst_json));
                console.log('Got CSTs',cst_json)
                let cst_results = cst_json['sentences']
                let req = create_reformulate_request(ep_results, cst_results)
                dispatch(requestReformulationResult(req));
                return fetch('/reformulate/', { method: "POST",
                                                redirect: "follow", body: JSON.stringify({
                                                    sentences: req
                                                })
                                              })
            })
            .then( (response) => {
                if (response.ok) return response.json();
                else return response.text().then(text => {throw Error(text)});
            })
            .then( (json) => {
                dispatch(receiveReformulationResult(json));
                console.log('Got reformulated CSTs',json)
                dispatch(receiveImprovedCSTResult(json));
            })
            .catch( (error) => dispatch(showApiError(error)) );
    }
}
