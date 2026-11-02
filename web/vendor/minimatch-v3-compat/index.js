'use strict'
// minimatch 3 exported a callable function; 11 exports named members. Keep the old shape, run the patched code.
const m = require('minimatch-patched')

function minimatch (path, pattern, options) {
  return m.minimatch(path, pattern, options)
}

Object.assign(minimatch, m)
minimatch.minimatch = minimatch
module.exports = minimatch
