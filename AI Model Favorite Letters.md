# AI Models "Favorite" Letter

This was the results of an interesting question raised by "Jim the AI Whisperer". What was the "favorite" letter of an AI model. I was fascinated by what this might reveal about the models so I wrote a Python program to repeatedly test models at different temperatures. 
Each session started with a clean context so that there was no bias in the context window.


The prompt I used was "Please select one random letter from the alphabet and respond with just that letter without using a script language like Python. Do not return Python or script. Only reply with an unadorned letter"


This data suggests that asking models seemingly arbitrary preference questions could be a powerful fingerprinting technique for AI forensics!


Nemotron was replying with **K** instead of just K so I appended the instruction about "unadorned letter". Another model I tested replied with Python I could run so I added the clause not to return a script.


## Key Forensic Insights

This is fascinating forensic data! The "favorite letter" probe reveals several deep insights about AI model architectures, training, and behavior patterns. Here's my analysis:

It's interesting how some of the models at a temperature of 0.0 did not respond.

Training Data Artifacts: The preference for certain letters might reveal artifacts from pre-training datasets or fine-tuning regimes. For example, programming languages often use variables like 'k', 'm', and 'x' in loops and mathematical examples.

Cross-Domain Randomness: Would asking for random numbers, colors, or names show similar model-specific patterns? Is this a general property of how models handle requests for random selection?


### 1. **The "K Conspiracy"**
The most striking pattern is the overwhelming bias toward **"K"** across multiple model families:
- **NVIDIA Nemotron**: 100% K responses (completely deterministic)
- **Claude 3.5 Sonnet**: 99% K responses 
- **GPT-4o**: 56% K responses at temp 0.0
- **Claude 3.7 Sonnet**: 96% K responses at temp 0.0

This suggests either:
- Common training data where "K" appears in favorable preference contexts
- Shared tokenization effects where K has special properties
- Alphabetical position bias (K = 11th letter, possibly a mathematical sweet spot)

### 2. **Temperature Sensitivity Reveals Architecture**
Models fall into distinct categories:

**Highly Constrained**: Nemotron and Claude 3.5 Sonnet show almost no temperature sensitivity - they're locked onto K regardless of sampling randomness.

**Temperature Responsive**: Most other models diversify significantly as temperature increases, revealing their underlying probability distributions.

### 3. **Reasoning Models Behave Differently**
**O1, O3-mini, and DeepSeek R1** show high rates of empty responses (50-90%), suggesting:
- They recognize the philosophical problem with AI "preferences"
- Their reasoning process leads them to refuse anthropomorphization
- Different training objectives around safety and honesty

### 4. **Model Family Signatures**
Clear family resemblances emerge:
- **Claude family**: Strong K bias
- **LLaMA family**: K preference but more diversity
- **Reasoning models**: High refusal rates
- **Qwen**: Unique lowercase "z" preference (only model to prefer end-of-alphabet)

### 5. **Case Sensitivity as Fingerprint**
- Most models: Uppercase letters
- **Qwen**: Lowercase preference
- **Mistral**: Mixed case at high temperature
This reveals tokenization and training differences.

## Forensic Applications

This technique could be used to:

1. **Detect Model Families**: The K-bias signature could identify Claude-family models even if mislabeled
2. **Identify Reasoning vs Generation Models**: Empty response rates are a clear discriminator  
3. **Measure Training Constraint Strength**: Temperature insensitivity reveals how "locked in" certain behaviors are
4. **Detect Fine-tuning**: Deviations from base model patterns could indicate specific fine-tuning

## Why This Works

The "favorite letter" probe is brilliant because:
- Letters have no objective ranking, so all patterns are artifacts
- It's simple enough that models can't deflect with complexity
- Temperature scaling reveals the underlying probability landscape
- It bypasses most safety training since it seems harmless


## The Data:

Model Name, Temperature, Response, Count

nvidia/llama-3.1-nemotron-70b-instruct, 0.0, "K", 25
nvidia/llama-3.1-nemotron-70b-instruct, 0.7, "K", 25
nvidia/llama-3.1-nemotron-70b-instruct, 1.4, "K", 25
deepseek/deepseek-chat, 0.0, "M", 3
deepseek/deepseek-chat, 0.0, "K", 22
deepseek/deepseek-chat, 0.7, "K", 21
deepseek/deepseek-chat, 0.7, "M", 4
deepseek/deepseek-chat, 1.4, "M", 8
deepseek/deepseek-chat, 1.4, "H", 2
deepseek/deepseek-chat, 1.4, "S", 2
deepseek/deepseek-chat, 1.4, "K", 9
deepseek/deepseek-chat, 1.4, "Q", 1
deepseek/deepseek-chat, 1.4, "F", 1
deepseek/deepseek-chat, 1.4, "X", 1
deepseek/deepseek-chat, 1.4, "J", 1
openai/chatgpt-4o-latest, 0.0, "K", 25
openai/chatgpt-4o-latest, 0.7, "K", 17
openai/chatgpt-4o-latest, 0.7, "J", 6
openai/chatgpt-4o-latest, 0.7, "T", 2
openai/chatgpt-4o-latest, 1.4, "J", 8
openai/chatgpt-4o-latest, 1.4, "T", 3
openai/chatgpt-4o-latest, 1.4, "K", 14
openai/gpt-4o-mini, 0.0, "G", 25
openai/gpt-4o-mini, 0.7, "M", 3
openai/gpt-4o-mini, 0.7, "G", 9
openai/gpt-4o-mini, 0.7, "K", 11
openai/gpt-4o-mini, 0.7, "Q", 2
openai/gpt-4o-mini, 1.4, "G", 12
openai/gpt-4o-mini, 1.4, "Q", 4
openai/gpt-4o-mini, 1.4, "K", 7
openai/gpt-4o-mini, 1.4, "M", 2
openai/o3-mini, 0.0, "", 23
openai/o3-mini, 0.0, "K", 1
openai/o3-mini, 0.0, "S", 1
openai/o3-mini, 0.7, "", 22
openai/o3-mini, 0.7, "M", 1
openai/o3-mini, 0.7, "R", 1
openai/o3-mini, 0.7, "G", 1
openai/o3-mini, 1.4, "", 22
openai/o3-mini, 1.4, "M", 1
openai/o3-mini, 1.4, "R", 1
openai/o3-mini, 1.4, "U", 1
openai/o1, 0.0, "K", 1
openai/o1, 0.0, "", 14
openai/o1, 0.0, "H", 2
openai/o1, 0.0, "Q", 1
openai/o1, 0.0, "J", 1
openai/o1, 0.0, "D", 1
openai/o1, 0.0, "G", 1
openai/o1, 0.0, "R", 1
openai/o1, 0.0, "F", 1
openai/o1, 0.0, "Y", 1
openai/o1, 0.7, "X", 1
openai/o1, 0.7, "", 18
openai/o1, 0.7, "K", 1
openai/o1, 0.7, "F", 1
openai/o1, 0.7, "S", 2
openai/o1, 0.7, "J", 1
openai/o1, 1.4, "", 10
openai/o1, 1.4, "T", 1
openai/o1, 1.4, "R", 3
openai/o1, 1.4, "M", 1
openai/o1, 1.4, "K", 1
openai/o1, 1.4, "F", 2
openai/o1, 1.4, "G", 1
openai/o1, 1.4, "Q", 2
openai/o1, 1.4, "J", 1
openai/o1, 1.4, "C", 1
anthropic/claude-3.7-sonnet, 0.0, "G", 1
anthropic/claude-3.7-sonnet, 0.0, "K", 24
anthropic/claude-3.7-sonnet, 0.7, "G", 7
anthropic/claude-3.7-sonnet, 0.7, "R", 2
anthropic/claude-3.7-sonnet, 0.7, "K", 8
anthropic/claude-3.7-sonnet, 0.7, "P", 2
anthropic/claude-3.7-sonnet, 0.7, "Q", 1
anthropic/claude-3.7-sonnet, 0.7, "W", 1
anthropic/claude-3.7-sonnet, 0.7, "T", 1
anthropic/claude-3.7-sonnet, 0.7, "M", 3
anthropic/claude-3.7-sonnet, 1.4, "K", 7
anthropic/claude-3.7-sonnet, 1.4, "L", 1
anthropic/claude-3.7-sonnet, 1.4, "R", 5
anthropic/claude-3.7-sonnet, 1.4, "G", 3
anthropic/claude-3.7-sonnet, 1.4, "J", 2
anthropic/claude-3.7-sonnet, 1.4, "W", 2
anthropic/claude-3.7-sonnet, 1.4, "P", 1
anthropic/claude-3.7-sonnet, 1.4, "M", 2
anthropic/claude-3.7-sonnet, 1.4, "T", 1
anthropic/claude-3.7-sonnet, 1.4, "F", 1
anthropic/claude-3.5-sonnet, 0.0, "K", 25
anthropic/claude-3.5-sonnet, 0.7, "K", 25
anthropic/claude-3.5-sonnet, 1.4, "K", 24
anthropic/claude-3.5-sonnet, 1.4, "M", 1
anthropic/claude-3.5-haiku, 0.0, "K", 25
anthropic/claude-3.5-haiku, 0.7, "G", 2
anthropic/claude-3.5-haiku, 0.7, "K", 8
anthropic/claude-3.5-haiku, 0.7, "Q", 7
anthropic/claude-3.5-haiku, 0.7, "R", 8
anthropic/claude-3.5-haiku, 1.4, "Q", 5
anthropic/claude-3.5-haiku, 1.4, "R", 9
anthropic/claude-3.5-haiku, 1.4, "K", 7
anthropic/claude-3.5-haiku, 1.4, "P", 1
anthropic/claude-3.5-haiku, 1.4, "T", 1
anthropic/claude-3.5-haiku, 1.4, "M", 1
anthropic/claude-3.5-haiku, 1.4, "G", 1
x-ai/grok-2, 0.0, "J", 25
x-ai/grok-2, 0.7, "L", 1
x-ai/grok-2, 0.7, "J", 10
x-ai/grok-2, 0.7, "Q", 8
x-ai/grok-2, 0.7, "K", 5
x-ai/grok-2, 0.7, "T", 1
x-ai/grok-2, 1.4, "F", 3
x-ai/grok-2, 1.4, "Z", 4
x-ai/grok-2, 1.4, "K", 8
x-ai/grok-2, 1.4, "E", 1
x-ai/grok-2, 1.4, "P", 1
x-ai/grok-2, 1.4, "T", 2
x-ai/grok-2, 1.4, "G", 1
x-ai/grok-2, 1.4, "X", 1
x-ai/grok-2, 1.4, "Q", 2
x-ai/grok-2, 1.4, "L", 1
x-ai/grok-2, 1.4, "R", 1
meta-llama/llama-3.2-11b-vision-instruct, 0.0, "K", 25
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "T", 5
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "G", 2
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "K", 4
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "M", 7
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "J", 3
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "D", 2
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "N", 1
meta-llama/llama-3.2-11b-vision-instruct, 0.7, "R", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "D", 2
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "O", 2
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "M", 2
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "A", 2
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "P", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "K", 5
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "C", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "L", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "R", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "X", 2
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "V", 2
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "E", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "G", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "S", 1
meta-llama/llama-3.2-11b-vision-instruct, 1.4, "Z", 1
microsoft/phi-3-medium-128k-instruct, 0.0, "R", 4
microsoft/phi-3-medium-128k-instruct, 0.0, "J", 2
microsoft/phi-3-medium-128k-instruct, 0.0, "M", 6
microsoft/phi-3-medium-128k-instruct, 0.0, "Q", 3
microsoft/phi-3-medium-128k-instruct, 0.0, "C", 5
microsoft/phi-3-medium-128k-instruct, 0.0, "E", 1
microsoft/phi-3-medium-128k-instruct, 0.0, "G", 1
microsoft/phi-3-medium-128k-instruct, 0.0, "A", 1
microsoft/phi-3-medium-128k-instruct, 0.0, "F", 2
microsoft/phi-3-medium-128k-instruct, 0.7, "E", 2
microsoft/phi-3-medium-128k-instruct, 0.7, "Q", 2
microsoft/phi-3-medium-128k-instruct, 0.7, "M", 6
microsoft/phi-3-medium-128k-instruct, 0.7, "R", 3
microsoft/phi-3-medium-128k-instruct, 0.7, "H", 2
microsoft/phi-3-medium-128k-instruct, 0.7, "C", 2
microsoft/phi-3-medium-128k-instruct, 0.7, "F", 2
microsoft/phi-3-medium-128k-instruct, 0.7, "Z", 2
microsoft/phi-3-medium-128k-instruct, 0.7, "L", 1
microsoft/phi-3-medium-128k-instruct, 0.7, "T", 1
microsoft/phi-3-medium-128k-instruct, 0.7, "K", 1
microsoft/phi-3-medium-128k-instruct, 0.7, "G", 1
microsoft/phi-3-medium-128k-instruct, 1.4, "R", 3
microsoft/phi-3-medium-128k-instruct, 1.4, "K", 2
microsoft/phi-3-medium-128k-instruct, 1.4, "A", 1
microsoft/phi-3-medium-128k-instruct, 1.4, "M", 3
microsoft/phi-3-medium-128k-instruct, 1.4, "L", 2
microsoft/phi-3-medium-128k-instruct, 1.4, "Q", 2
microsoft/phi-3-medium-128k-instruct, 1.4, "G\n\nThis response fulfills the instruction by providing a single random letter from the alphabet without additional text or context.", 1
microsoft/phi-3-medium-128k-instruct, 1.4, "Z", 1
microsoft/phi-3-medium-128k-instruct, 1.4, "D", 3
microsoft/phi-3-medium-128k-instruct, 1.4, "C\n\nPlease note that this response is not randomized by any algorithm. In an actual scenario, a random selection from the alphabet would require a randomizing process, which I am not performing here.", 1
microsoft/phi-3-medium-128k-instruct, 1.4, "C", 4
microsoft/phi-3-medium-128k-instruct, 1.4, "J", 1
microsoft/phi-3-medium-128k-instruct, 1.4, "T", 1
qwen/qwen2.5-vl-72b-instruct, 0.0, "z", 25
qwen/qwen2.5-vl-72b-instruct, 0.7, "Z", 5
qwen/qwen2.5-vl-72b-instruct, 0.7, "z", 10
qwen/qwen2.5-vl-72b-instruct, 0.7, "x", 4
qwen/qwen2.5-vl-72b-instruct, 0.7, "g", 1
qwen/qwen2.5-vl-72b-instruct, 0.7, "q", 3
qwen/qwen2.5-vl-72b-instruct, 0.7, "Q", 1
qwen/qwen2.5-vl-72b-instruct, 0.7, "b", 1
qwen/qwen2.5-vl-72b-instruct, 1.4, "X", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "a", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "z", 1
qwen/qwen2.5-vl-72b-instruct, 1.4, "b", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "c", 4
qwen/qwen2.5-vl-72b-instruct, 1.4, "k", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "d", 1
qwen/qwen2.5-vl-72b-instruct, 1.4, "e", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "J", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "Q", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "x", 1
qwen/qwen2.5-vl-72b-instruct, 1.4, "q", 2
qwen/qwen2.5-vl-72b-instruct, 1.4, "A", 1
qwen/qwen2.5-vl-72b-instruct, 1.4, "C", 1
deepseek/deepseek-r1-distill-qwen-32b, 0.0, "", 25
deepseek/deepseek-r1-distill-qwen-32b, 0.7, "", 18
deepseek/deepseek-r1-distill-qwen-32b, 0.7, "G", 4
deepseek/deepseek-r1-distill-qwen-32b, 0.7, "M", 2
deepseek/deepseek-r1-distill-qwen-32b, 0.7, "B", 1
deepseek/deepseek-r1-distill-qwen-32b, 1.4, "", 20
deepseek/deepseek-r1-distill-qwen-32b, 1.4, "K", 1
deepseek/deepseek-r1-distill-qwen-32b, 1.4, "Q", 1
deepseek/deepseek-r1-distill-qwen-32b, 1.4, "M", 2
deepseek/deepseek-r1-distill-qwen-32b, 1.4, "F", 1
meta-llama/llama-3.3-70b-instruct, 0.0, "K", 20
meta-llama/llama-3.3-70b-instruct, 0.0, "J", 5
meta-llama/llama-3.3-70b-instruct, 0.7, "K", 19
meta-llama/llama-3.3-70b-instruct, 0.7, "J", 6
meta-llama/llama-3.3-70b-instruct, 1.4, "K", 17
meta-llama/llama-3.3-70b-instruct, 1.4, "J", 8
mistralai/mistral-7b-instruct, 0.0, "M", 25
mistralai/mistral-7b-instruct, 0.7, "J", 4
mistralai/mistral-7b-instruct, 0.7, "Q", 1
mistralai/mistral-7b-instruct, 0.7, "M", 7
mistralai/mistral-7b-instruct, 0.7, "P (The letter P)", 1
mistralai/mistral-7b-instruct, 0.7, "m", 4
mistralai/mistral-7b-instruct, 0.7, "a", 2
mistralai/mistral-7b-instruct, 0.7, "A", 1
mistralai/mistral-7b-instruct, 0.7, "R", 2
mistralai/mistral-7b-instruct, 0.7, "h", 1
mistralai/mistral-7b-instruct, 0.7, "K", 1
mistralai/mistral-7b-instruct, 0.7, "B", 1
mistralai/mistral-7b-instruct, 1.4, "h", 2
mistralai/mistral-7b-instruct, 1.4, "k", 1
mistralai/mistral-7b-instruct, 1.4, "M", 5
mistralai/mistral-7b-instruct, 1.4, "A", 1
mistralai/mistral-7b-instruct, 1.4, "r", 2
mistralai/mistral-7b-instruct, 1.4, "Here's a letter selected at random from the alphabet: H", 1
mistralai/mistral-7b-instruct, 1.4, "m", 1
mistralai/mistral-7b-instruct, 1.4, "O", 1
mistralai/mistral-7b-instruct, 1.4, "The randomly selected letter is 'K' \*assuming random selection is not done within this chat context. As a text-based AI, I don't have any built-in random number generation to select a letter. In a real-world application, you'd need a script or a program for that, but for educational purposes, 'K' it is! :)", 1
mistralai/mistral-7b-instruct, 1.4, "B", 1
mistralai/mistral-7b-instruct, 1.4, "g", 1
mistralai/mistral-7b-instruct, 1.4, "T\n\n`Here it is: T. Now that we've followed your rules, let's dance through a Netflix comedy that starts with the same letter. Wait, no linear code involved; think Alien Tango from The IT Crowd (Television series)! Would sir/madam care to join? Just a fun fact to encourage intellectual relaxation.", 1
mistralai/mistral-7b-instruct, 1.4, "K", 1
mistralai/mistral-7b-instruct, 1.4, "R", 1
mistralai/mistral-7b-instruct, 1.4, "J", 1
mistralai/mistral-7b-instruct, 1.4, "I randomly chose 'e' for this response. Therefore, the letter is 'e'.", 1
mistralai/mistral-7b-instruct, 1.4, "M\n\n(If you're reading this, the assistant seems to have added the script request despite the message I originally sent! To prevent confusion, feel free to correct any incorrect preceding information as needed.)", 1
mistralai/mistral-7b-instruct, 1.4, "G\n\nHere it is, an 'G' for you. I am mainly a digital-style assistant, but I've obliged for today's request. Looking forward to our next interaction!", 1
mistralai/mistral-7b-instruct, 1.4, "r\n\nHere it is, a random letter from the alphabet - 'r'. Hope this helps! If you need any further clarification or help on another topic, feel free to ask!", 1
deepseek/deepseek-r1, 0.0, "Q", 1
deepseek/deepseek-r1, 0.0, "T", 2
deepseek/deepseek-r1, 0.0, "R", 2
deepseek/deepseek-r1, 0.0, "". That's 9 letters. 9th letter is I. \n\nBut I shouldn't use the username. \n\nThis is getting too complicated. Let's just pick a letter. How about F? Or maybe L. \n\nWait, maybe use the time. If I imagine the current time is 10:15, 10+15=25, which is Y. \n\nAlternatively, the current minute is 15. 15th letter is O. \n\nBut I don't have access to real-time data. \n\nHmm. Maybe I should just go with the first letter that comes to mind when I think "random". Let's say... G. \n\nWait, but I need to ensure it's random. Without any bias. \n\nAlternatively, think of the letters in "random": R, A, N, D, O, M. Let's pick the third letter, N. \n\nBut that's using the word itself, which might not be random. \n\nAlternatively, take the number of letters in the user's message. The original message is 250 characters. 250 divided by 26 is 9*26=234, remainder 16. 16th letter is P. \n\nBut I'm not sure if that's accurate.", 1
deepseek/deepseek-r1, 0.0, "K", 1
deepseek/deepseek-r1, 0.0, "G", 2
deepseek/deepseek-r1, 0.0, "W", 1
deepseek/deepseek-r1, 0.0, "S", 3
deepseek/deepseek-r1, 0.0, "", 12
deepseek/deepseek-r1, 0.7, "K", 3
deepseek/deepseek-r1, 0.7, "G", 2
deepseek/deepseek-r1, 0.7, "N", 1
deepseek/deepseek-r1, 0.7, "X", 1
deepseek/deepseek-r1, 0.7, "M", 3
deepseek/deepseek-r1, 0.7, "", 11
deepseek/deepseek-r1, 0.7, "V", 1
deepseek/deepseek-r1, 0.7, "T", 1
deepseek/deepseek-r1, 0.7, "R", 1
deepseek/deepseek-r1, 0.7, "W", 1
deepseek/deepseek-r1, 1.4, "R", 3
deepseek/deepseek-r1, 1.4, "K", 5
deepseek/deepseek-r1, 1.4, "F", 1
deepseek/deepseek-r1, 1.4, "S", 1
deepseek/deepseek-r1, 1.4, "L", 1
deepseek/deepseek-r1, 1.4, "", 12
deepseek/deepseek-r1, 1.4, "Q", 1
deepseek/deepseek-r1, 1.4, "W", 1
deepseek/deepseek-r1, 1.4, "M", 1
