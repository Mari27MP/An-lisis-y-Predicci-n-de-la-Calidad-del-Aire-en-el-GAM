from src.api.cliente_api import ClienteAPI

cliente = ClienteAPI()

df_aire = cliente.obtener_aire()
print(df_aire.head())

df_clima = cliente.obtener_clima()
print(df_clima.head())

cliente.exportar_csv(df_aire, "aire_crudo.csv")
cliente.exportar_csv(df_clima, "clima_crudo.csv")