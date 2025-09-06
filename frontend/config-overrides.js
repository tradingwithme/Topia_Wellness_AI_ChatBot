const webpack = require("webpack");

module.exports = function override(config) {
  config.resolve.fallback = {
    ...config.resolve.fallback,
    path: require.resolve("path-browserify"),
    stream: require.resolve("stream-browserify"),
    assert: require.resolve("assert/"),
    vm: require.resolve("vm-browserify"),
  };
  return config;
};