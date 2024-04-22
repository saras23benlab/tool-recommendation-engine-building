from pydomo import Domo
import pandas as pd

api_host = 'api.domo.com'
client_id= ${{ secrets.DOMO_CLIENT}}
secret_id = $ {{ secrets.DOMO_SECRET}}
domo= Domo(client_id,secret_id,api_host)
dataset_id= 'b6002499-e7ab-4aa2-8617-3e07f90a0be5'
payments_data = domo.ds_get('b6002499-e7ab-4aa2-8617-3e07f90a0be5')