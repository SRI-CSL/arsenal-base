// This is used to make redux-logger output readable
// immutable objects.
import Immutable from 'immutable';

function transform(state) {
  if (state === null){
    return state;
  }
  const t = typeof(state);
  if (t === 'number' || t === 'boolean' || t === 'string' || t === 'undefined') {
    return state;
  }
  if (state instanceof Array){
    return state.map(el => transform(el));
  }
  if (Immutable.Iterable.isIterable(state)) {
    return state.toJS();
  }
  let newState = {};
  for (let i of Object.keys(state)) {
    newState[i] = transform(state[i]);
  }
  return newState;
}

export default transform;
