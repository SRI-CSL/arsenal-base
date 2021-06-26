//import { combineReducers } from 'redux';
import * as Actions from '../actions/ActionTypes';

export const initialState = {
  nl: '',
  entities: null,
  cst: null,
  improvedCst: null,
  cstIsHidden: true,
  cstIsLoading: false,
  entitiesIsHidden: true,
  entitiesIsLoading: false,
  nlIsLoading: false,
  apiError: null
}

function rootReducer(state = initialState, action) {
  switch (action.type) {
    case Actions.SHOW_API_ERROR: return {...state, apiError: action.error };
    case Actions.HIDE_API_ERROR: return {...state, apiError: null };
    case Actions.REQUEST_ENTITY_RESULT: return {...state, entities: null, regexes: '', entitiesIsLoading: true };
    case Actions.RECEIVE_ENTITY_RESULT: return {...state, entities: action.result, entitiesIsLoading: false };
    case Actions.REQUEST_CST_RESULT: return {...state, cst: null, cstIsLoading: true };
    case Actions.RECEIVE_CST_RESULT: return {...state, cst: action.result, cstIsLoading: false };
    case Actions.SET_NL: return {...state, nl: action.text }
    case Actions.SET_REGEXES: return {...state, regexes: action.text }
    case Actions.REQUEST_IMPROVED_CST_RESULT: return {...state, improvedCst: null, improvedCstIsLoading: true };
    case Actions.RECEIVE_IMPROVED_CST_RESULT: return {...state, improvedCst: action.result, improvedCstIsLoading: false };
    case Actions.HIDE_CST: return {...state, cstIsHidden: true };
    case Actions.SHOW_CST: return {...state, cstIsHidden: false };
    case Actions.HIDE_ENTITIES: return {...state, entitiesIsHidden: true };
    case Actions.SHOW_ENTITIES: return {...state, entitiesIsHidden: false };
    default: return state;
  }


}

export default rootReducer;
