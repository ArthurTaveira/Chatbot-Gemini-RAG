# Chatbot-Gemini-RAG
Eis aqui um chatbot, uma aplicação Q&amp;A que irá utilizar o LLM do google, Gemini, e uma Técnica chamada de RAG
# Q&A com RAG e GEMINI

**Motivation** : Large Language Models can reason about many topics they were trained on. The motivation is to augment yout level of knowledge to help with questions outside of your scope of knowledge. This can be done with RAG (Retrivel Augmented generation )

**Motivação:** Modelos de Linguagem de Grande Escala (LLMs) podem raciocinar sobre muitos tópicos nos quais foram treinados. A motivação é aumentar o seu nível de conhecimento para ajudar com perguntas fora do seu escopo de conhecimento. Isso pode ser feito com **RAG** (Geração Aumentada por Recuperação)

**Dependencias**:

langchain
langchain_community
langchain-google-genai
python-dotenv
streamlit
langchain_experimental
sentence-transformers
langchain_chroma
langchainhub
pypdf
rapidocr-onnxruntime
unstructured

**Carregamento e processamento de dados**

1º Etapa:

 Importação dos métodos e das bibliotecas necessárias para o projeto, a  medida que forem sendo usadas, seus fins serão explicados. É importante ressaltar que não é um projeto absoluto e que podem ser usados outros llm’s, outros processos de Embeddings e outras formas de criação de um ambiente virtual.

```python
import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader

```

2º Etapa: 

Importando a chave API do Gemini do Google por meio de uma biblioteca python chamada de dotenv, utilizada para manipular de forma segura variáveis de ambiente em um arquivo .env

a estrutura do arquivo deve ser esta:

GOOGLE_API_KEY="xxxxxxxxxx”

`st.title()` foi usado apenas definir um titulo no ambiente virtual.

```python

st.title("RAG Application built on Gemini Model")
from dotenv import load_dotenv
load_dotenv()
```

3º Etapa:

define-se uma variável urls, composta por uma lista de strings que são urls dos sites, é possível colocar mais de um site dentro do array, basta separa-los por virgula.

usa-se o método UnstructuredURLLoader(armazenado numa variável loader, que serve para buscar conteúdo da Web a partir de URLs, como parametro foi passado a variável criada com o mesmo nome do parâmetro.

logo após, em uma variável data, carrega-se os dados.

```python
urls = ['https://www.correiobraziliense.com.br/politica/2024/11/6995204-mp-tcu-pede-suspensao-dos-salarios-de-bolsonaro-e-outros-militares.html']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
```

4º Etapa: 

utiliza-se `RecursiveCharacterTextSplitter`() para dividir o documento em partes menores, por meio de um parâmetro chunk_size, stabele-se que o tamanho do chunk será de 1000 caracteres.

e é importante destacar que é um processo recursivo, feito de forma hierarquica, ideal para textos com subcamadas, como por exemplo, parágrafos.

depois, em docs, utiliza-se o text_splitte e por meio do split_document(), recebe o do conjunto de dados carregados e os divide como definido no text_splitte

isso é útil para evitar problemas em textos grandes, isto permite buscas mais rapidas e precisas e mais eficiência no processamento

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

```

**Criação de um Vectorstore usando Chroma:**

5º Etapa: 

Este conjunto de código limpa o cache do chomadb, evitando complicações.

```
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()
```

6º Etapa:

Chorma é uma biblioteca de busca e armazenamento de Embeddings, que permite criar um banco de dados vetoriais, a partir de documentos baseados em similaridade

A função **`Chroma.from_documents()`** recebe documentos, os docs criado no processamento de dados e o modelo de linguagem, neste caso o `GoogleGenerativeAIEmbeddings`. 

a IA trabalhará representações vetoriais e não com palavras

`retriever` as_retriver é uma função que busca em um vectorstore, pedações de documentos por similaridade semântica, e o `search_kwargs` serve para limitar a quantidade de documentos retornados com a query (busca)

```python
vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
```

7º Etapa: 

`ChatGoogleGenerativeAI` é uma classe do langchain que permite acessar modelos de linguagem feitos pelo google, como o Gemini

parâmetros:

model - Define qual llm será usado

temperature - pode-se usar valores entre 0 a 1, quanto mais proximo de 1, a resposta será mais criativa em e menos determinística, sendo o valor mais determinístico o 0

max_tokens - determina a quantidade de tokens de resposta que o llm pode gerar como resposta

timeout - define o tempo para que o modelo gere a resposta

```jsx
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_tokens=None,timeout=None)
```

8º Etapa:

Criação da Querry (pergunta) para ser passada para o llm. Aqui, usa-se uma query que foi pessada pelo client(pessoa utilizando o programa), por meio de uma interface amigável

```jsx
query = st.chat_input("Say something: ") 
```

mas para outros fins, a query pode ser passada manualmente afim de não utilizar a interface.

**Predação do Modelo:**

9ª Etapa: 

predação do modelo:

definindo o `system_prompt`, ele permite definir o comportamento do modelo, permitindo vasta diferenciação, no fim, é destinado o {contexto}, que será destinado ao contexto gerado pelo RAG

e o prompt, usa-se o `ChatPromptTemplate.from_messages`, e dentro de um arrya, organiza-se em tuplas, o comportamento do sistema, que recebe todas as regras de orientação definidas anteriormente, e o human, como um input, que vem da query

```jsx
system_prompt = (
    "Você é um assistente para tarefas de resposta a perguntas."
    "Use as seguintes partes do contexto recuperado para responder "
    "a pergunta. Se você não sabe a resposta, diga que você não sabe"
    "Mantenha a resposta concisa."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

```

etapa 10º: 

**`create_stuff_documents_chain`** é uma função do Langchain que cria uma **cadeia de processamento** (chain) para realizar a tarefa de **responder a perguntas** a partir de documentos
nela, são passados o llm e o prompt

**`create_retrieval_chain`** é uma função que cria uma cadeia de recuperação de informações, integrando o **retriever** (responsável por recuperar os documentos relevantes) e a **questão/resposta** (criada na etapa anterior)
são passados o retriver, ja criado anteriormente e o question_answer_chain

**`rag_chain.invoke()`** é o método que **executa a cadeia de recuperação e resposta**.

nele são colocados como o input a query que foi digitada pelo client

e por fim essa resposta é exibida no ambiente virtual.

```jsx
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])
```

para executar o aplicativo, execute o streamlit com `streamlit run app.py app.py` (ou o nome do seu arquivo que você criou) em um ambiente de execução

Confira o Artigo no Medium: https://medium.com/@arthurguedes001/chatbot-com-rag-retrieval-augmented-generation-usando-o-llm-gemini-54070d830202

Isto esta sendo uma fase do meu processo como Pesquisador do CNPq, espero estar disseminando mais conhecimento em portgues sobre o assunto, e qualquer dúvida ou melhoria que queira adicionar, fico a disposição, pode mandar uma PL ou um email (arthurguedes001@gmail.com) Abraço

Aqui está o video de referencia usado para o código:
https://www.youtube.com/watch?v=8cKf5GUz4TU&t=180s
