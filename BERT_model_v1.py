 # Load model 

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2") 

 # Define your texts FIRST 

requirement = "The telescope shall maintain thermal stability within ±0.1°C during observations." 

rule = "Open-ended, non-verifiable terms such as provide support, but not limited to, or as \a minimum" 

 

 # Encode 

req_emb = model.encode(requirement, convert_to_tensor=True) 

rule_emb = model.encode(rule, convert_to_tensor=True) 

 

 # Similarity score 

score = util.cos_sim(req_emb, rule_emb).item() 

print(f"Similarity: {score:.3f}") 

Similarity: 0.203 

 

 # Keyword-based heuristic 

OPEN_ENDED_PHRASES = [ 

     "provide a choice", 

         "as required", 

             "as appropriate", 

                 "where necessary", 

                     "support", 

                         "but not limited to" 

                         ] 

 

req_lower = requirement.lower() 

 

keyword_flag = any(p in req_lower for p in OPEN_ENDED_PHRASES) 

 #final decision 

if keyword_flag or score <= 0.6: 

     print("likelyviolates open-ended / non-verifiable rule") 

else: 

     print("likely compliant") 