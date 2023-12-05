import React, { Component } from 'react';
import './App.css';
import ES6Promise from 'es6-promise';
import createStoreWithMiddleware from './store/configureStore';
import { Provider } from 'react-redux';
import rootReducer, { initialState } from './reducers';
import version from './version';
import Home from './components/Home';

// Note: This is read from package.json, so make sure to update
// that before building, so we can keep track of which build is
// running
console.log('Arsenal version', version);

// import logo from './logo.svg';
// import './App.css';

// Promise polyfill for IE
ES6Promise.polyfill();

const store = createStoreWithMiddleware(rootReducer, initialState);

class App extends Component {
  render() {
    return (
      <Provider store={store}>
        <Home/>
      </Provider>
    );
  }
}

export default App;
