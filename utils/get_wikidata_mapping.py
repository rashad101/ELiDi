import requests
# Get the wikidata mapping for a freebase entity


def get_wikidata_mapping(fb_id):
    """
    Given the freebase id, get the wikidata mapping
    """
    url = 'https://query.wikidata.org/sparql'
    query = """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/> 
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX v: <http://www.wikidata.org/prop/statement/>
        
        SELECT ?item ?itemLabel WHERE {
           ?item wdt:P646 """+fb_id+""".
          SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en" .
           }
        }
    """
    try:
        r = requests.get(url, params={'format': 'json', 'query': query})
        data = r.json()
        print("get_wikidata_mapping -> json, ", data)
        wiki_map = data['results']['bindings'][0]['item']['value']  # getting the mapping
    except Exception:
        wiki_map = ''
    return wiki_map


def fetch_entity(entity):
    url_template = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&" \
                   "language=en&type=item&continue=0&search=ENTITY"
    try:
        wiki_ent_search_url = url_template.replace("ENTITY", '%20'.join(entity.split()))
        data = requests.get(url=wiki_ent_search_url).json()
        wiki_url = data['search'][0]['url']
        wiki_id = data["search"][0]["id"]
        wiki_label = data["search"][0]["label"]
    except Exception:
        wiki_url, wiki_id, wiki_label = '','',''
    return wiki_url, wiki_id,wiki_label


if __name__ == '__main__':
    wiki_map = get_wikidata_mapping("'/m/02sj6'")
    print (wiki_map)
