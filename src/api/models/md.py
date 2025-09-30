import re

class MetaData(object):

    @property
    def has_metadata(self):
        return '"="' in self._prompt

    def __init__(self, prompt: str):
        self._prompt = prompt
    
    def get_metadata_args(self):
        outer_key = "orAll"
        md_args = {outer_key: []}

        pattern = r'"([^"]*)"\s*=\s*"([^"]*)"' # TODO: DRY on pattern
        matches = re.findall(pattern, self._prompt)

        for k,v in dict(matches).items():
            sub_map = {"equals": {"key": k, "value": v}}
            md_args[outer_key].append(sub_map)
        
        # Can't have andAll with just one filter :(
        if len(matches) == 1:
            md_args = md_args[outer_key][0]

        return md_args
    
    def get_clean_query(self):
        return re.sub(r'"[^"]*"\s*=\s*"[^"]*"', '', self._prompt).strip()

if __name__ == "__main__":
    md = MetaData('"OE_Number"="111" Tell me about the event.')
    prompt = md.get_clean_query()
    filters = md.get_metadata_args()

    print(f"Prompt: {prompt}\nFilters: {filters}")