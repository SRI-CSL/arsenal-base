//import { combineReducers } from 'redux';
import * as Actions from '../actions/ActionTypes';

export const initialState = {
  nl: '',
  entities: null,
  entities_string: '',
  cst: null,
  improvedCst: null,
  cstIsHidden: true,
  cstIsLoading: false,
  reformulationIsLoading: false,
  improvedCstIsLoading: false,
  entitiesIsHidden: true,
  entitiesIsLoading: false,
  nlIsLoading: false,
  apiError: null,
  noopEntity: false
}

function rootReducer(state = initialState, action) {
  switch (action.type) {
    case Actions.SHOW_API_ERROR: return {...state, apiError: action.error };
    case Actions.HIDE_API_ERROR: return {...state, apiError: null };
    case Actions.CLEAR_RESULTS: return {...state, entities: null, cst: null, improvedCst: null };
    case Actions.REQUEST_ENTITY_RESULT: return {...state, entities: null, regexes: '', entitiesIsLoading: true };
    case Actions.RECEIVE_ENTITY_RESULT: return {...state, entities: action.result, entitiesIsLoading: false, entities_string: JSON.stringify(action.result, null, '\t') };
    case Actions.REQUEST_CST_RESULT: return {...state, cst: null, cstIsLoading: true };
    case Actions.RECEIVE_CST_RESULT: return {...state, cst: action.result, cstIsLoading: false };
    case Actions.REQUEST_REFORMULATION_RESULT: return {...state, reformulationIsLoading: true };
    case Actions.RECEIVE_REFORMULATION_RESULT: return {...state, reformulationIsLoading: false };
    case Actions.REQUEST_IMPROVED_CST_RESULT: return {...state, improvedCst: null, improvedCstIsLoading: true };
    case Actions.RECEIVE_IMPROVED_CST_RESULT: return {...state, improvedCst: action.result, improvedCstIsLoading: false };
    case Actions.SET_NL: return {...state, nl: action.text }
    case Actions.SET_ENTITIES: return {...state, entities_string: action.text }
    case Actions.HIDE_CST: return {...state, cstIsHidden: true };
    case Actions.SHOW_CST: return {...state, cstIsHidden: false };
    case Actions.HIDE_ENTITIES: return {...state, entitiesIsHidden: true };
    case Actions.SHOW_ENTITIES: return {...state, entitiesIsHidden: false };
    case Actions.NOOP_ENTITY_PROCESSOR: return {...state, noopEntity: true };
    case Actions.REAL_ENTITY_PROCESSOR: return {...state, noopEntity: false };
    default: return state;
  }
}

export default rootReducer;
