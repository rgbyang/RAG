## Overview

- A basic LLM application as knowledge base. To help you to understand the basic workflow and funcitons.
- You can have the LLM answer your questions from the context you provide.
- Main steps: vectorization (embedding), RAG.

## Environment

- OS: Windows.
- Hardware: GPU is not required.
- Software: Python and the related packages.

## Getting Started

1. Run "Vectorization.py" to vectorize your context and save it to local "embeddingsBert.npy" and "chunksBert.txt".
2. Run "RAG.py" to load "embeddingsBert.npy" and "chunksBert.txt" to memory, and output the anwswer to the terminal.
3. You can specify your context in "context.py", and specify your quesiton by changing the value of "question" in "RAG.py".

## License

This project is licensed under the Apache License 2.0. 

### Third-Party Licenses

This project uses the following third-party libraries:

- **Library transformers**: Licensed under the Apache License 2.0
- **Library numpy**: Licensed under the BSD License
- **Library torch**: Licensed under the BSD License
- **Library faiss**: Licensed under the MIT License
- **Library colorama**: Licensed under the BSD License

For more details, see the [LICENSE](LICENSE) file.
