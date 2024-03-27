export const isReadableStream = (obj: any): obj is ReadableStream => {
	return obj && typeof obj.getReader === 'function';
};
