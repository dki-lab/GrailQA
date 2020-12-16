'use strict';
const BinBuild = require('bin-build');
const log = require('logalot');
const bin = require('.');

bin.run(['--help'], err => {
	if (err) {
		log.warn(err.message);
		log.warn('zopflipng pre-build test failed');
		log.info('compiling from source');

		let makeBin = 'make';
		let makeArgs = '';

		if (process.platform === 'freebsd') {
			makeBin = 'gmake';
			makeArgs = 'CC=cc CXX=c++';
		}

		const builder = new BinBuild()
			.src('https://github.com/google/zopfli/archive/64c6f362fefd56dccbf31906fdb3e31f6a6faf80.zip')
			.cmd(`mkdir -p ${bin.dest()}`)
			.cmd(`${makeBin} zopflipng ${makeArgs} && mv ./zopflipng ${bin.path()}`);

		return builder.run(err => {
			if (err) {
				log.error(err.stack);
				return;
			}

			log.success('zopflipng built successfully');
		});
	}

	log.success('zopflipng pre-build test passed successfully');
});
