//以下所有 buffer 都是所有元素都是 Number 的 Array，每个元素表示一个字节。

function pushVuint(buffer, val){
	for(var i = 0; i < 6; ++i){
		var by = val & 0x7F;
		val >>>= 7;
		if(val == 0){
			buffer.push(by);
			return;
		}
		buffer.push(by | 0x80);
	}
	buffer.push(val & 0xFF);
}
function pushVint(buffer, val){
	if(val >= 0){
		pushVuint(buffer, val * 2);
	} else {
		pushVuint(buffer, (-val + 1) * 2);
	}
}
function pushRaw(buffer, val){ // val = [ 0, 1, 2, 3 ]
	for(var i = 0; i < val.length; ++i){
		buffer.push(val[i]);
	}
}
function pushBytes(buffer, val){
	pushVuint(buffer, val.length);
	pushRaw(buffer, val);
}
function pushString(buffer, val){
	var bytes = new Array;
	for(var i = 0; i < val.length; ++i){
		var codePoint = val.charCodeAt(i);
		if((0xD800 <= codePoint) && (codePoint < 0xDC00)){
			var leading = codePoint & 0x3FF;
			var trailing = val.charCodeAt(++i) & 0x3FF;
			codePoint = ((leading << 10) | trailing) + 0x10000;
		}
		if(codePoint < 0x80){ // 7 位
			bytes.push(codePoint);
		} else if(codePoint < 0x800){ // 11 位 = 5 + 6
			bytes.push(((codePoint >>>  6) & 0x1F) | 0xC0);
			bytes.push(((codePoint       ) & 0x3F) | 0x80);
		} else if(codePoint < 0x10000){ // 16 位 = 4 + 6 + 6
			bytes.push(((codePoint >>> 12) & 0x0F) | 0xE0);
			bytes.push(((codePoint >>>  6) & 0x3F) | 0x80);
			bytes.push(((codePoint       ) & 0x3F) | 0x80);
		} else { // 21 位 = 3 + 6 + 6 + 6
			bytes.push(((codePoint >>> 18) & 0x07) | 0xF0);
			bytes.push(((codePoint >>> 12) & 0x3F) | 0x80);
			bytes.push(((codePoint >>>  6) & 0x3F) | 0x80);
			bytes.push(((codePoint       ) & 0x3F) | 0x80);
		}
	}
	pushBytes(buffer, bytes);
}

function shiftVuint(buffer){
	var val = 0;
	for(var i = 0; i < 6; ++i){
		var by = buffer.shift();
		val |= (by & 0x7F) << (7 * i);
		if(!(by & 0x80)){
			return val;
		}
	}
	val <<= 8;
	val |= buffer.shift() & 0xFF;
	return val;
}
function shiftVint(buffer){
	var tmp = shiftVuint(buffer);
	if(!(tmp & 1)){
		return tmp / 2;
	} else {
		return -(tmp + 1) / 2;
	}
}
function shiftRaw(buffer, length){
	var val = new Array;
	for(var i = 0; i < length; ++i){
		val.push(buffer.shift() & 0xFF);
	}
	return val;
}
function shiftBytes(buffer){
	var length = shiftVuint(buffer);
	return shiftRaw(buffer, length);
}
function shiftString(buffer){
	var val = new String();
	var bytes = shiftBytes(buffer);
	while(bytes.length > 0){
		var codePoint = bytes.shift();
		if(codePoint >= 0x80){
			if((codePoint & 0xE0) == 0xC0){ // 11 位
				codePoint &= 0x1F;
				codePoint <<= 6;
				codePoint |=  bytes.shift() & 0x3F;
			} else if((codePoint & 0xF0) == 0xE0){ // 16 位
				codePoint &= 0x0F;
				codePoint <<= 12;
				codePoint |= (bytes.shift() & 0x3F) <<  6;
				codePoint |=  bytes.shift() & 0x3F;
			} else { // 21 位
				codePoint &= 0x0F;
				codePoint <<= 18;
				codePoint |= (bytes.shift() & 0x3F) << 12;
				codePoint |= (bytes.shift() & 0x3F) <<  6;
				codePoint |=  bytes.shift() & 0x3F;
			}
		}

		if(codePoint < 0x10000){
			val += String.fromCharCode(codePoint);
		} else {
			var leading = ((codePoint - 0x10000) >>> 10) & 0x3FF;
			var trailing = codePoint & 0x3FF;
			val += String.fromCharCode(leading | 0xD800, trailing | 0xDC00);
		}
	}
	return val;
}

// socket 是一个 WebSocket 对象。
function sendBytes(socket, buffer){
	var arrayBuffer = new ArrayBuffer(buffer.length);
	var view = new Uint8Array(arrayBuffer);
	for(var i = 0; i < buffer.length; ++i){
		view[i] = buffer[i];
	}
	socket.send(arrayBuffer);
}
function recvBytes(blob, callback){
	var reader = new FileReader();
	reader.onloadend = function(){
		if(!reader.result){
			// TODO
			throw "Error receiving bytes";
		}

		var view = new Uint8Array(reader.result);
		var buffer = new Array;
		for(var i = 0; i < view.length; ++i){
			buffer.push(view[i]);
		}
		callback(buffer);
	};
	reader.readAsArrayBuffer(blob);
}
