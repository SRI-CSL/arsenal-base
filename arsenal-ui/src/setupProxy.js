const proxy = require('http-proxy-middleware');
 
module.exports = function(app) {
  // Entity proxy
  app.use(proxy('/entity', {
    target: 'http://localhost:8060/',
    pathRewrite: {
      '^/entity': ''
    }
  }))

  // Noop-Entity proxy
  app.use(proxy('/noop_entity', {
    target: 'http://localhost:8061/',
    pathRewrite: {
      '^/noop_entity': ''
    }
  }))
  
  // Cleaner proxy
  app.use(proxy('/cleaner', {
    target: 'http://localhost:8069/',
    pathRewrite: {
      '^/cleaner': ''
    }
  }))
  
  // NLP2CST proxy
  app.use(proxy('/nl2cst', {
    target: 'http://localhost:8070/',
    pathRewrite: {
      '^/nl2cst': ''
    }
  }))

  // Reformulate proxy
  app.use(proxy('/reformulate', {
    target: 'http://localhost:8090/',
    pathRewrite: {
      '^/reformulate': ''
    }
  }))
  
}