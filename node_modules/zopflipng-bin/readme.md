# zopflipng-bin [![Build Status](https://travis-ci.org/imagemin/zopflipng-bin.svg?branch=master)](https://travis-ci.org/imagemin/zopflipng-bin)

> [zopfli](https://github.com/google/zopfli) Compression Algorithm is a new zlib (gzip, deflate) compatible compressor that takes more time (~100x slower), but compresses around 5% better than zlib and better than any other zlib-compatible compressor

You probably want [`imagemin-zopfli`](https://github.com/imagemin/imagemin-zopfli) instead.


## Install

```
$ npm install zopflipng-bin
```


## Usage

```js
const {execFile} = require('child_process');
const zopflipng = require('zopflipng-bin');

execFile(zopflipng, ['-m', '--lossy_8bit', 'input.png', 'outout.png'], () => {
	console.log('Image minified!');
});
```


## CLI

```
$ npm install --global zopflipng-bin
```

```
$ zopflipng --help
```


## License

MIT © [Imagemin](https://github.com/imagemin)
