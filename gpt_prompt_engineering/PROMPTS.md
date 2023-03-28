# Sequential Prompting
While ChatGPT struggles to perform the task of relation extraction given a single, comprehensive prompt, we are able to split this task up into smaller, sequential sub-tasks to enhance performance.

## Prompt Formatting
We will be providing our prompts with examples in order to facilitate few shot learning with examples from our training data. We are following these [gudelines](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions) from Microsoft. The `prompts.json` file in this directory provides example formatting, where all user/assistant interactions before the final user query are few-shot learning examples. In order to be compatible with `run_prompts.py`, the final dictionary in the list for any given prompt should be the only one that needs text formatting to be applied on a given abstract; the rest of the prompts should have all text explicitly filled in in order to function as examples.

## Prompt Sequencing
The prompts should be written such that the current prompt's output is the next prompt's input.
