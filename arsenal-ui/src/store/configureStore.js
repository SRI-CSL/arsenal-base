import { createStore, applyMiddleware } from 'redux';
import { createLogger } from 'redux-logger';
import thunk from 'redux-thunk';
import { composeWithDevTools } from 'redux-devtools-extension';
import immutableTransform from '../util/ImmutableLogTransformer';

// See this for options:
// https://github.com/fcomb/redux-logger
const loggerMiddleware = createLogger({
  collapsed: true,
  stateTransformer: immutableTransform,
  actionTransformer: immutableTransform
});

export const createStoreWithMiddleware = (rootReducer, initialState={}) => createStore(
  rootReducer,
  initialState,
  composeWithDevTools(
    applyMiddleware(
      thunk, // lets us dispatch() functions
      loggerMiddleware // neat middleware that logs actions
    )
  )
);

export default createStoreWithMiddleware;