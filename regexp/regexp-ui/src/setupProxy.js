const proxy = require('http-proxy-middleware');
 
module.exports = function(app) {
  // Entity proxy
  app.use(proxy('/entity', {
    target: 'http://localhost:8060/',
    pathRewrite: {
      '^/entity': ''
    }
  }))

  // NLP2CST proxy
  app.use(proxy('/nl2cst', {
    target: 'http://localhost:8070/',
    pathRewrite: {
      '^/nl2cst': ''
    }
  }))

}