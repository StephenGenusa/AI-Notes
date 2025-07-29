# Prompting - Prompt Optimization Formats

## Natural Language Prompts

Natural language prompts (English expressions) have been found to be useful for flexible, reasoning-heavy tasks, creative generation, and exploratory queries where nuance and human-like conversation are key. They excel in scenarios requiring adaptability but can lead to ambiguity without structure. Research from 2024-2025 shows they often serve as a strong baseline, with performance variations up to 40% compared to structured formats, particularly outperforming in complex reasoning [1][2][3][4].

| # | Category | Prompt |
|---|----------|--------|
| 1 | Consumer | Suggest a healthy meal plan for a week, including breakfast, lunch, and dinner options that are easy to prepare. |
| 2 | Programmer Technical | Explain how to implement a binary search algorithm in Python, step by step, with code examples. |
| 3 | Electrical Distribution | Describe the key components of a smart grid system and how they improve energy efficiency. |
| 4 | Consumer | Recommend budget-friendly home workout routines for beginners aiming to lose weight. |
| 5 | Programmer Technical | Debug this JavaScript code that's causing an infinite loop and suggest fixes. |
| 6 | Electrical Distribution | Outline safety protocols for installing high-voltage transformers in urban areas. |
| 7 | Consumer | Provide tips on how to organize a small apartment to maximize space. |
| 8 | Programmer Technical | Compare the pros and cons of using React vs. Vue for front-end development. |
| 9 | Electrical Distribution | Explain how load balancing works in electrical distribution networks to prevent blackouts. |
| 10 | Consumer | Give advice on choosing the best smartphone under $500 with good battery life. |
| 11 | Programmer Technical | Write a SQL query to find the top 10 customers by total purchase amount from a sales database. |
| 12 | Electrical Distribution | Discuss common causes of power surges and methods to protect household appliances. |
| 13 | Consumer | Create a packing list for a 7-day beach vacation in summer. |
| 14 | Programmer Technical | Describe best practices for securing API endpoints in a Node.js application. |
| 15 | Electrical Distribution | Summarize the role of substations in converting high-voltage to low-voltage for residential use. |

Citations for natural language prompts include:  
[1] https://arxiv.org/html/2411.10541v1  
[2] https://arxiv.org/html/2505.20139v1  
[3] https://www.researchgate.net/publication/382885370_Let_Me_Speak_Freely_A_Study_on_the_Impact_of_Format_Restrictions_on_Performance_of_Large_Language_Models  
[4] https://aclanthology.org/2024.naacl-long.429/

## JSON Prompts

JSON prompts are useful for structured data extraction, API-like interactions, parameter-heavy tasks, and ensuring parsable outputs in applications. They provide precision and are widely supported, though verbose, making them ideal for classification and schema-enforced responses. 2025 research highlights JSON's strength in non-renderable outputs but notes it's less efficient than YAML for token consumption [5][6][7][8].

| # | Category | Prompt |
|---|----------|--------|
| 1 | Consumer | {"task": "recommend_recipes", "ingredients": ["chicken", "rice", "vegetables"], "diet": "low-carb", "servings": 4} |
| 2 | Programmer Technical | {"action": "generate_code", "language": "Python", "function": "sort_list", "input": "unsorted array", "output_format": "function"} |
| 3 | Electrical Distribution | {"query": "analyze_grid", "components": ["transformers", "cables"], "focus": "efficiency_metrics", "units": "kWh"} |
| 4 | Consumer | {"request": "travel_plan", "destination": "Paris", "duration": "5 days", "budget": "mid-range", "interests": ["sightseeing", "food"]} |
| 5 | Programmer Technical | {"debug": true, "code_snippet": "for i in range(10): print(i)", "error": "IndexError", "language": "Java"} |
| 6 | Electrical Distribution | {"simulation": "power_flow", "network_type": "radial", "load": 5000, "voltage": 11000, "output": "stability_report"} |
| 7 | Consumer | {"advice": "fitness", "goal": "build_muscle", "level": "intermediate", "equipment": "home_gym"} |
| 8 | Programmer Technical | {"compare": ["Docker", "Kubernetes"], "criteria": ["scalability", "ease_of_use"], "context": "cloud_deployment"} |
| 9 | Electrical Distribution | {"troubleshoot": "outage", "symptoms": ["flickering_lights", "low_voltage"], "system": "residential_grid"} |
| 10 | Consumer | {"product_recommend": "laptop", "specs": {"RAM": "16GB", "storage": "512GB"}, "price_max": 1000} |
| 11 | Programmer Technical | {"query_sql": "SELECT * FROM users WHERE age > 30", "optimize": true, "database": "PostgreSQL"} |
| 12 | Electrical Distribution | {"design": "wiring_system", "building_type": "commercial", "capacity": "100kVA", "safety_standards": "IEC"} |
| 13 | Consumer | {"list": "grocery", "diet": "vegan", "meals": 7, "include": "prices_estimate"} |
| 14 | Programmer Technical | {"secure_api": "best_practices", "framework": "Express.js", "features": ["authentication", "rate_limiting"]} |
| 15 | Electrical Distribution | {"calculate": "energy_loss", "line_length": 10, "current": 200, "resistance": 0.05, "units": "km_A_ohm"} |

Citations for JSON prompts include:  
[5] https://arxiv.org/html/2505.20139v1  
[6] https://medium.com/@tahirbalarabe2/%EF%B8%8Fstructured-output-in-llms-why-json-xml-format-matters-c644a81cf4f3  
[7] https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df  
[8] https://github.com/imaurer/awesome-llm-json

## YAML Prompts

YAML prompts are useful for configurations, workflows, multi-step processes, and token-efficient structured inputs due to their concise, human-readable indentation. They outperform JSON in efficiency and cost for language models, making them suitable for large-scale applications and chained tasks. Recent studies emphasize YAML's advantages in readability and performance for LLM outputs [9][10][11][12].

| # | Category | Prompt |
|---|----------|--------|
| 1 | Consumer | task: meal_planning<br>days: 7<br>preferences:<br>  - vegetarian<br>  - quick_prep |
| 2 | Programmer Technical | generate:<br>  language: C++<br>  type: class<br>  name: Vector<br>  methods:<br>    - add<br>    - remove |
| 3 | Electrical Distribution | analyze:<br>  system: substation<br>  parameters:<br>    voltage: 220kV<br>    load: peak |
| 4 | Consumer | workout:<br>  type: yoga<br>  duration: 30min<br>  level: beginner<br>  focus: flexibility |
| 5 | Programmer Technical | debug:<br>  code: while(true) {}<br>  language: Rust<br>  issue: memory_leak |
| 6 | Electrical Distribution | safety_check:<br>  equipment: cables<br>  standards: OSHA<br>  environment: underground |
| 7 | Consumer | organize:<br>  space: kitchen<br>  items:<br>    - utensils<br>    - appliances<br>  goal: efficiency |
| 8 | Programmer Technical | framework_comparison:<br>  options:<br>    - Flask<br>    - Django<br>  metrics: speed, scalability |
| 9 | Electrical Distribution | balance_load:<br>  network: three_phase<br>  demand: 10MW<br>  strategy: dynamic |
| 10 | Consumer | gadget_recommend:<br>  category: earbuds<br>  features:<br>    - wireless<br>    - noise_cancelling<br>  budget: 100 |
| 11 | Programmer Technical | db_query:<br>  type: MongoDB<br>  operation: aggregate<br>  collection: orders<br>  filter: date > 2025-01-01 |
| 12 | Electrical Distribution | surge_protection:<br>  devices: home_appliances<br>  methods:<br>    - SPD<br>    - grounding |
| 13 | Consumer | vacation_pack:<br>  destination: mountains<br>  season: winter<br>  duration: 5_days |
| 14 | Programmer Technical | api_security:<br>  protocol: HTTPS<br>  tools:<br>    - JWT<br>    - OAuth |
| 15 | Electrical Distribution | convert_voltage:<br>  input: 33kV<br>  output: 400V<br>  efficiency: calculate |

Citations for YAML prompts include:  
[9] https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df  
[10] https://www.preprints.org/manuscript/202506.1937/v1  
[11] https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1558938/full  
[12] https://arxiv.org/html/2411.10541v1

## XML Prompts

XML prompts are useful for semantic reinforcement through tagging, complex hierarchical structures, and tasks requiring explicit context separation, such as reasoning chains or annotated data. They aid in high-performance tasks and interpretability, often outperforming plain text in structured reasoning for models like GPT-4. 2025 analyses show XML's value in reliable, maintainable LLM designs despite higher verbosity [13][14][15][16].

| # | Category | Prompt |
|---|----------|--------|
| 1 | Consumer | <prompt><task>budget_shopping</task><items><item>groceries</item><item>clothes</item></items><limit>200</limit></prompt> |
| 2 | Programmer Technical | <code><language>Go</language><function>concurrency</function><example>goroutines</example></code> |
| 3 | Electrical Distribution | <analysis><grid>urban</grid><elements><element>switches</element><element>relays</element></elements></analysis> |
| 4 | Consumer | <health><plan>diet</plan><goals><goal>weight_loss</goal><goal>energy_boost</goal></goals></health> |
| 5 | Programmer Technical | <debug><snippet>int x = 0 / 0;</snippet><error>division_by_zero</error><fix>suggest</fix></debug> |
| 6 | Electrical Distribution | <protocol><installation>overhead_lines</installation><safety><rule>insulation</rule></safety></protocol> |
| 7 | Consumer | <home><organization>bedroom</organization><tips><tip>storage</tip><tip>lighting</tip></tips></home> |
| 8 | Programmer Technical | <comparison><frameworks><fw>Angular</fw><fw>Svelte</fw></frameworks><aspects>performance</aspects></comparison> |
| 9 | Electrical Distribution | <network><type>distribution</type><issues><issue>overloading</issue></issues><solutions>upgrade</solutions></network> |
| 10 | Consumer | <recommend><product>smartwatch</product><features><feature>fitness_tracking</feature></features><price>under_150</price></recommend> |
| 11 | Programmer Technical | <query><db>Oracle</db><type>join</type><tables><table>employees</table><table>departments</table></tables></query> |
| 12 | Electrical Distribution | <protection><event>fault</event><devices><device>circuit_breaker</device></devices></protection> |
| 13 | Consumer | <travel><list>essentials</list><trip>cruise</trip><duration>10_days</duration></travel> |
| 14 | Programmer Technical | <security><app>mobile</app><practices><practice>encryption</practice></practices></security> |
| 15 | Electrical Distribution | <substation><role>step_down</role><specs><spec>capacity_50MVA</spec></specs></substation> |

Citations for XML prompts include:  
[13] https://medium.com/@tahirbalarabe2/%EF%B8%8Fstructured-output-in-llms-why-json-xml-format-matters-c644a81cf4f3  
[14] https://www.linkedin.com/pulse/understanding-prompt-formats-xml-markdown-yaml-made-simple-paluy-fgtkc  
[15] https://community.openai.com/t/xml-vs-markdown-for-high-performance-tasks/1260014  
[16] https://news.ycombinator.com/item?id=40396857

## Markdown Prompts

Markdown prompts, as an additional lightweight format, are useful for organized, readable content with minimal overhead, such as lists, headings, and simple hierarchies. They convey intent effectively, boost performance in content generation, and are LLM-friendly for quick overviews or formatted responses. 2025 research indicates Markdown's efficiency advantages over complex formats like JSON/XML, especially in structured yet flexible tasks [17][18][19][20].

| # | Category | Prompt |
|---|----------|--------|
| 1 | Consumer | # Recipe Ideas<br>- Ingredients: eggs, bread, cheese<br>- Style: breakfast<br>- Time: under 15 min |
| 2 | Programmer Technical | ## Algorithm Guide<br>1. Input: array<br>2. Output: sorted<br>* Language: Swift |
| 3 | Electrical Distribution | # Grid Optimization<br>- Factors: demand, supply<br>- Metrics: loss percentage |
| 4 | Consumer | # Wellness Tips<br>- Activity: meditation<br>- Benefits: stress reduction |
| 5 | Programmer Technical | ## Error Fixing<br>Code: `print(1/0)`<br>Issue: ZeroDivisionError |
| 6 | Electrical Distribution | # Installation Steps<br>1. Check voltage<br>2. Connect wires<br>* Safety: wear gloves |
| 7 | Consumer | # Decor Ideas<br>- Room: living<br>- Theme: minimalist |
| 8 | Programmer Technical | ## Tool Comparison<br>- VS Code vs. IntelliJ<br>- For: Java development |
| 9 | Electrical Distribution | # Maintenance Checklist<br>- Inspect: insulators<br>- Frequency: monthly |
| 10 | Consumer | # Gift Suggestions<br>- Occasion: birthday<br>- Recipient: tech enthusiast<br>- Budget: $50 |
| 11 | Programmer Technical | ## Data Query<br>Table: products<br>Filter: price > 100 |
| 12 | Electrical Distribution | # Risk Assessment<br>- Hazard: arc flash<br>- Prevention: PPE |
| 13 | Consumer | # Event Planning<br>- Type: party<br>- Guests: 20<br>- Menu: simple |
| 14 | Programmer Technical | ## Code Review<br>Best Practices: comments, modularity |
| 15 | Electrical Distribution | # Energy Audit<br>- Building: factory<br>- Focus: lighting efficiency |

Citations for Markdown prompts include:  
[17] https://www.robertodiasduarte.com.br/en/markdown-vs-xml-em-prompts-para-llms-uma-analise-comparativa/  
[18] https://www.linkedin.com/pulse/understanding-prompt-formats-xml-markdown-yaml-made-simple-paluy-fgtkc  
[19] https://arxiv.org/html/2411.10541v1  
[20] https://news.ycombinator.com/item?id=40395057