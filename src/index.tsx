import { Ai } from '@cloudflare/ai';
import { Hono } from 'hono';
import { isReadableStream } from './utils';

// Connections to the different Cloudflare services
type Bindings = {
	AI: Ai;
	DB: D1Database;
	KNOWLEDGE_VECTOR: VectorizeIndex;
};

// Database Schema
type NoteRow = {
	id: string;
	text: string;
};

const app = new Hono<{ Bindings: Bindings }>();

app.get('/ask', async (c) => {
	const ai = new Ai(c.env.AI);

	// Get the question as a search parameter
	// Example - /?text=What is the capital of France?
	const question = c.req.query('question');

	if (!question) {
		return c.text('Missing text', 400);
	}

	// Get the vector embeddings for the question
	const embeddings = await ai.run('@cf/baai/bge-base-en-v1.5', { text: question });
	const vectors = embeddings.data[0];

	// Get the most similar note to the question
	const SIMILARITY_CUTOFF = 0.75;
	const vectorQuery = await c.env.KNOWLEDGE_VECTOR.query(vectors, { topK: 1 });
	const vecIds = vectorQuery.matches.filter((vec) => vec.score > SIMILARITY_CUTOFF).map((vec) => vec.id);

	// Get the notes for the most similar vectors
	let notes: string[] = [];
	if (vecIds.length) {
		const query = `SELECT * FROM notes WHERE id IN (${vecIds.join(', ')})`;
		const { results } = await c.env.DB.prepare(query).bind().all<NoteRow>();
		if (results) notes = results.map((vec) => vec.text);
	}

	// Context message
	const contextMessage = notes.length ? `Context:\n${notes.map((note) => `- ${note}`).join('\n')}` : '';

	// System prompt
	const systemPrompt = `When answering the question or responding, use the context provided, if it is provided and relevant.`;

	// Run the AI model to get the response
	const aiResponse = await ai.run('@cf/meta/llama-2-7b-chat-int8', {
		messages: [
			...(notes.length ? [{ role: 'system', content: contextMessage }] : []),
			{ role: 'system', content: systemPrompt },
			{ role: 'user', content: question },
		],
	});

	// If the response is aa stream, return
	if (isReadableStream(aiResponse)) {
		return;
	}

	return c.text(aiResponse.response ?? 'No response');
});

app.post('/notes', async (c) => {
	const ai = new Ai(c.env.AI);

	// Get the text from the request body
	// Example - { "text": "The capital of France is Paris." }
	const { text } = await c.req.json();
	if (!text) {
		return c.text('Missing text', 400);
	}

	// Insert the note into the database
	const { results } = await c.env.DB.prepare('INSERT INTO notes (text) VALUES (?) RETURNING *').bind(text).all<NoteRow>();

	const record = results.length ? results[0] : null;

	if (!record) {
		return c.text('Failed to create note', 500);
	}

	// Generate the vector embeddings for the note
	const { data } = await ai.run('@cf/baai/bge-base-en-v1.5', { text: [text] });
	const values = data[0];

	if (!values) {
		return c.text('Failed to generate vector embedding', 500);
	}

	// Insert the vector embeddings into the vector index
	const { id } = record;
	const inserted = await c.env.KNOWLEDGE_VECTOR.upsert([
		{
			id: id.toString(),
			values,
		},
	]);

	return c.json({ id, text, inserted });
});

app.onError((err, c) => {
	return c.text(err.message);
});

export default app;
