'use strict';
const path = require('path');
const BinWrapper = require('bin-wrapper');
const pkg = require('../package.json');

const url = `https://raw.github.com/imagemin/zopflipng-bin/v${pkg.version}/vendor/`;

module.exports = new BinWrapper()
	.src(`${url}osx/zopflipng`, 'darwin')
	.src(`${url}linux/zopflipng`, 'linux')
	.src(`${url}win32/zopflipng.exe`, 'win32')
	.dest(path.resolve(__dirname, '../vendor'))
	.use(process.platform === 'win32' ? 'zopflipng.exe' : 'zopflipng');
