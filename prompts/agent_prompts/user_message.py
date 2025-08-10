USER_MESSAGE = """
<|start|>user<|message|># Regex Golf Task

You will receive two lists. Produce the shortest regex (under the given grammar) that **fullmatches** all YES strings and none of the NO strings. Return ONLY the `regex_solution` object.

```json
{
  "yes": {yes_list},
  "no":  {no_list}
}
```<|end|>
"""