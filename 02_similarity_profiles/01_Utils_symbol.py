import requests
import sys
from tqdm import tqdm

def ensembl_id_to_gene(ensembl_id):
    
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("display_name", 'Symbol not found')
    # .get(key, default_value = None)
    else:
        return "NULL"

def gene_to_ensembl_id(symbol):
    url = f'https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{symbol}'
    r = requests.get(url, headers={ "Content-Type" : "application/json"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()
        return "NULL"
    else:
        decoded = r.json()
        try:
            return decoded[0]['id']
        except:
            return "NULL"

def main():

    geneset_path = f'/home/jhpark/page-reproducibility/02_similarity_profiles/intermediate/UnTangled_ensembleset.txt'
    save_path = f'/home/jhpark/page-reproducibility/02_similarity_profiles/intermediate/UnTangled_geneset.txt'
    converted_symbol = []
    
    with open(geneset_path, 'r') as f:
        for line in tqdm(f):
            result = ensembl_id_to_gene(line.strip('\n'))
            converted_symbol.append(result)
    with open(save_path, 'w') as f:
        for item in converted_symbol:
            f.write(item+'\n')

if __name__ == '__main__':
    main()