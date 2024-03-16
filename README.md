# chatAI

## Setup

### OpenAI token

In order to use the chatAI project you need an OpenAI token.
To create such a token go the OpenAI website [here](https://platform.openai.com/overview)

Then create a `.env` file at the root of the repository containing the token and the desired
OpenAI model to use for the chat:

```bash
OPENAI_API_KEY=<token>
OPENAI_MODEL=gpt-3.5-turbo #or gpt-4
```

### Documents

Create a directory called `Documents` at the root of the repository.
You may add all the documents you wish to get indexed to that folder.
It is recommended to create a flat folder structure instead of dumping
all documents in one place.

Your documents tree should look something like this:

```text
-- Documents
  -- Topic 1
    -- pdf1.pdf
       ...
  -- Topic 2
    -- pdf2.pdf
       ...
  -- Topic 3
    -- pdf3.pdf
       ...
```

## Running

After you are done with the setup just run the main file.
You can then update the PDFs, add new ones or delete them from the 
Documents folder. Every time you rerun the program it will detect
any changes you have made and update the indexed documents.

You may also specify a particular directory or file you wish to chat
with. Only the specified file or directory will be used as context in
that case.

NOTE: The required Pyhon version is 3.11.7

```commandline
python main.py -f Documents/Test/test.pdf
python main.py -d Documents/Test
```