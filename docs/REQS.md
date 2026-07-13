# REQS — Graph RAG Demo

## The idea

I would like to make a graph RAG pipeline. I want the pipeline to ingest text (like markdown files, txt files) and do named entity recognition, coreference resolution, and entity linking. At the end, I want the result to be a knowledge graph.

## Why / the problem

I am monitoring large amounts of data coming in from news sources, documents, etc. I am trying to identify connections between entities like places and people from these sources and try to look for patterns in the data. I need a knowledge graph that allows me to do multi-hop reasoning, connect disjointed documents, and find complex, indirect relationships.

## Who it's for

For myself, I am trying to showcase that I can ingest a large amount of data (assume it comes in as markdown or plain text) and build a knowledge graph out of it. I am a software engineer and trying to learn more about Graph RAG, so this is educational for me too. Ultimately, I want to be able to run the pipeline locally using Docker compose, and to benchmark the capability of this graph RAG pipeline using some benchmarks.

## What it should do

### Graph RAG pipeline 

- The pipeline should be triggered upon receiving a message from a Kafka message queue.
- The Kafka message will contain a bucket name and object key
- The first stage of the pipeline will read the markdown file / text file from S3 comptaible object storage (like MinIO) given the bucket name and object key from the Kafka message
- After that, the markdown file will go through a Named Entity Recognition (NER) stage. We will automatically identify and classify specific entities in unstructured text into predefined categories such as names of people, organizations, locations, dates, and quantities.
  - Example of input text:  "Elon Musk visited the Tesla factory in Berlin on March 12th."
  - NER Output: Elon Musk --> PERSON, Tesla --> ORGANIZATION, Berlin --> LOCATION, March 12th --> DATE
- After NER, we will perform Coreference Resolution. This is the task of finding all expressions (like pronouns) that refer to the same entity in a text. It connects the dots so an AI knows who or what "he", "she", "it", or "they" is talking about across multiple sentences.
  - Example input text: "Sarah went to the store. She bought some apples, but they were bruised."
  - Coref Resolution Output: "She" refers to Sarah; "they" refers to apples
- After that, we have an Entity Linking (EL) stage. Also known as Named Entity Disambiguation. Takes NER a step further. Once an entity is identified, EL maps it to its unique, definitive entry in a structured knowledge base (like Wikipedia or Wikidata). This resolves ambiguity when words have multiple meanings. For example:
  - Let's look at the word "Apple" in tow different contexts
  - Text A: "Apple launched the new iPhone today."
  - Text B: "I ate a crisp Apple for breakfast."
  - Entity Linking Output for Text A: Maps "Apple" to the Wikipedia URL https://en.wikipedia.org/wiki/Apple_Inc (organization)
  - Entity Linking Output for Text B: Maps "Apple" to https://en.wikipedia.org/wiki/Apple (Fruit).
- Finally, we have Graph Retrieval Augmented Generation (RAG). Standard RAG searches text chunks using vector similarity to help an LLM answer questions. Graph RAG takes this to the next level by structuring information into a Knowledge Graph (nodes and edges) before doing the retrieval. Instead of just finding matching words, it looks at how concepts are interconnected. This allows LLMs to reason about complex, multi-hop relationships across an entire document dataset. For example:
  - Imagine you upload 500 pages of corporate documents and ask: "How are our compliance risks in Germany tied to our supply chain suppliers?"
  - Standard RAG: Might pull up individual paragrphas mentioning "Germany" and "suppliers" separately, missing the big picture.
  - Graph RAG: Traverses the graph: `[Supplier A]` --> located in --> `[Berlin]` --> subject to --> `[German supply Chain Act]`. It synthesizes this connected path to give an accurate, holistic answer.
- For this, we will have a knowledge graph builder stage, that takes in the original file text and JSON of linked entities, and produce semantic triplets ready to be saved into Neo4j.

### External / OpenSource (that we will use but won't have to build from scratch)

The following services should be spun up in the same docker compose so that the document-graph-rag pipeline can access them:

- Kafka message queue
- MinIO object storage
- Elasticsearch (two instances / storages? - one for entities, one for Documents)
  - The original markdown / text document should be saved to Elasticsearch (Documents) before any processing
  - During the entity linking step, I am thinking of reading from Elasticsearch (Entities) first. I believe the output of entity linking will be a JSON of linked clusters; advise me on where I should save this new JSON (Elasticsearch Entities, or Documents, or both?).
  - The knowledge graph builder stage will read from Elasticsearch (please advise on Entities, Documents, etc.)
- Neo4J for the knowledge graph
  - The knowledge graph builder stage will write to Neo4J
- vLLM / external LLM API (ok to use ChatGPT 4o for this; I have an API key ready).
  - I expect coreference-resolution, entity-linking, and knowledge graph builder stages to all make use of the LLM API.

## Nice-to-haves (maybe)

- Frontend and service for visualizing the Elasticsearch Entities and Documents, as well as the Knowledge Graph.
- A way to balance benchmarking the performance of this pipeline without burning through LLM API credits (advice on this would be welcome)

## Explicitly NOT doing

- No authentication
- No self hosting of models; use external API for LLM.
- Supporting multi users / deploying this publicly (yet), I just want to be able to run the whole thing locally in docker compose.

## Constraints / notes

- Make sure the code is modular
- Use python 3.12 for this project
- Use uv to manage dependencies
- Use Svelte + Vite if building any frontend
- Follow coding best practices (docstrings for functions and classes, type hints, etc.)
