import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2 import service_account
from babel.numbers import format_currency
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
import decimal
import datetime
from google.cloud import secretmanager
import json
from google.cloud import bigquery
import requests

def puxar_aba_simples(id_gsheet, nome_aba, nome_df):

    nome_credencial = st.secrets["CREDENCIAL_SHEETS"]
    credentials = service_account.Credentials.from_service_account_info(nome_credencial)
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = credentials.with_scopes(scope)
    client = gspread.authorize(credentials)

    spreadsheet = client.open_by_key(id_gsheet)
    
    sheet = spreadsheet.worksheet(nome_aba)

    sheet_data = sheet.get_all_values()

    st.session_state[nome_df] = pd.DataFrame(sheet_data[1:], columns=sheet_data[0])

def tratar_colunas_numero_df(df, lista_colunas):

    for coluna in lista_colunas:

        try:

            df[coluna] = (df[coluna].str.replace('.', '', regex=False).str.replace(',', '.', regex=False))

            df[coluna] = pd.to_numeric(df[coluna])

        except:

            df[coluna] = pd.to_numeric(df[coluna])

def tratar_colunas_data_df(df, lista_colunas):

    for coluna in lista_colunas:

        df[coluna] = pd.to_datetime(df[coluna]).dt.date

def tratar_colunas_data_df_2(df, lista_colunas):

    for coluna in lista_colunas:

        df[coluna] = pd.to_datetime(df[coluna], format='%d/%m/%Y').dt.date

def adicionar_colunas_ano_mes(df, coluna_data=None):

    if not coluna_data is None:

        df['Ano'] = pd.to_datetime(df[coluna_data]).dt.year

        df['Mes'] = pd.to_datetime(df[coluna_data]).dt.month

        df['Mes_Ano'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mes'].astype(str) + '-01').dt.to_period('M')

    else:

        df['Mes_Ano'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mes'].astype(str) + '-01').dt.to_period('M')

    return df

def puxar_dados_gsheet(id_gsheet, nome_aba, nome_df, colunas_numero=None, colunas_data=None, coluna_data_ano_mes=None):

    puxar_aba_simples(
        id_gsheet, 
        nome_aba, 
        nome_df
    )

    if colunas_numero is None:

        tratar_colunas_numero_df(
            st.session_state[nome_df], 
            st.session_state[nome_df].columns
        )

    elif colunas_numero == 'Nenhuma':

        pass

    else:

        tratar_colunas_numero_df(
            st.session_state[nome_df], 
            colunas_numero
        )

    if not colunas_data is None:

        tratar_colunas_data_df(
            st.session_state[nome_df], 
            colunas_data
        )

    if coluna_data_ano_mes == 'Nenhuma':

        pass

    elif not coluna_data_ano_mes is None:

        st.session_state[nome_df] = adicionar_colunas_ano_mes(
            st.session_state[nome_df], 
            coluna_data_ano_mes
        )

    else:

        st.session_state[nome_df] = adicionar_colunas_ano_mes(
            st.session_state[nome_df]
        )

def criar_coluna_setor_definir_metas(df):

    if st.session_state.base_luck == 'test_phoenix_salvador':

        dict_setor = dict(zip(st.session_state.df_canal_de_vendas_setor['Canal de Vendas'], st.session_state.df_canal_de_vendas_setor['Setor']))

        df.loc[pd.notna(df['Canal_de_Vendas']), 'Setor'] = df['Canal_de_Vendas'].map(dict_setor)

    df_metas_indexed = st.session_state.df_metas.set_index('Mes_Ano')

    df['Meta'] = df.apply(lambda row: df_metas_indexed.at[row['Mes_Ano'], row['Setor']] 
                          if row['Setor'] in df_metas_indexed.columns and row['Mes_Ano'] in df_metas_indexed.index 
                          else 0, axis=1)
    
    return df

def puxar_df_historico(id_gsheet, nome_aba, nome_df):

    puxar_aba_simples(
        id_gsheet, 
        nome_aba, 
        nome_df
    )

    tratar_colunas_numero_df(
        st.session_state[nome_df], 
        st.session_state[nome_df].drop(columns=['Setor']).columns
    )

    st.session_state[nome_df] = adicionar_colunas_ano_mes(
        st.session_state[nome_df]
    )

    st.session_state[nome_df] = criar_coluna_setor_definir_metas(
                                    st.session_state[nome_df]
                                )

def puxar_df_campanha(id_gsheet, noma_aba_1, nome_df_1, colunas_numero_df_1, colunas_data_df_1, nome_aba_2, nome_df_2, coluna_data_ano_mes):

    puxar_aba_simples(
        id_gsheet, 
        noma_aba_1, 
        nome_df_1
    )

    tratar_colunas_numero_df(
        st.session_state[nome_df_1], 
        colunas_numero_df_1
    )

    tratar_colunas_data_df(
        st.session_state[nome_df_1], 
        colunas_data_df_1
    )

    puxar_aba_simples(
        id_gsheet, 
        nome_aba_2, 
        nome_df_2
    )

    st.session_state[nome_df_1] = st.session_state[nome_df_1].merge(
                                      st.session_state[nome_df_2], 
                                      left_on='Veículo', 
                                      right_on='Veiculo', 
                                      how='left'
                                  )

    st.session_state[nome_df_1] = st.session_state[nome_df_1][pd.notna(st.session_state[nome_df_1]['Data de Abastecimento'])].reset_index(drop=True)

    st.session_state[nome_df_1] = adicionar_colunas_ano_mes(
        st.session_state[nome_df_1], 
        coluna_data_ano_mes
    )   

def gerar_df_big_query_tratado():

    def puxar_df_big_query():

        project_id = "base-omie-analise"

        secret_id = "Cred"

        secret_client = secretmanager.SecretManagerServiceClient()

        secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

        response = secret_client.access_secret_version(request={"name": secret_name})

        secret_payload = response.payload.data.decode("UTF-8")

        credentials_info = json.loads(secret_payload)

        autenticar = service_account.Credentials.from_service_account_info(credentials_info)

        config_cliente = bigquery.Client(credentials=autenticar, project=autenticar.project_id)

        consulta_geral = f"""
        SELECT * 
        FROM `base-omie-analise.BD_Luck.Base_Omie` 
        """
        df = config_cliente.query(consulta_geral).to_dataframe()
        
        return df

    st.session_state.df_big_query = puxar_df_big_query()

    st.session_state.df_big_query = st.session_state.df_big_query[st.session_state.df_big_query['BD']!='Base_Kuara'].reset_index(drop=True)

    st.session_state.df_big_query = st.session_state.df_big_query[~st.session_state.df_big_query['Desc_Depto'].isin(['Mansear', 'Kuara'])].reset_index(drop=True)

def gerar_dict_categorias_alteradas():

    puxar_dados_gsheet(
        id_gsheet=st.session_state.id_gsheet_bd_omie_luck, 
        nome_aba='BD_Categorias_Alteradas', 
        nome_df='df_categorias_alteradas',
        colunas_numero='Nenhuma',
        colunas_data=None,
        coluna_data_ano_mes='Nenhuma'
    )

    st.session_state.dict_categorias_alteradas = dict(zip(st.session_state.df_categorias_alteradas['Categoria Anterior'], st.session_state.df_categorias_alteradas['Categoria Atual']))

def gerar_df_phoenix(base_luck, request_select):

    config = {
        'user': 'user_automation_jpa', 
        'password': 'luck_jpa_2024', 
        'host': 'comeia.cixat7j68g0n.us-east-1.rds.amazonaws.com', 
        'database': base_luck
        }

    conexao = mysql.connector.connect(**config)

    cursor = conexao.cursor()

    request_name = request_select

    cursor.execute(request_name)

    resultado = cursor.fetchall()
    
    cabecalho = [desc[0] for desc in cursor.description]

    cursor.close()

    conexao.close()

    df = pd.DataFrame(resultado, columns=cabecalho)

    df = df.applymap(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)

    return df

def gerar_df_vendas_final():

    def gerar_df_vendas_manuais(id_gsheet, nome_aba, nome_df, colunas_numero, colunas_data):

        puxar_aba_simples(id_gsheet, nome_aba, nome_df)

        tratar_colunas_numero_df(st.session_state[nome_df], colunas_numero)

        tratar_colunas_data_df_2(st.session_state[nome_df], colunas_data)

    def ajustar_nomes_leticia_soraya(df_vendas):

        df_vendas['Vendedor'] = df_vendas['Vendedor'].replace('SORAYA - TRANSFERISTA', 'SORAYA - GUIA')

        df_vendas.loc[df_vendas['Vendedor']=='SORAYA - GUIA', 'Setor'] = 'Guia'

        df_vendas.loc[(df_vendas['Vendedor']=='LETICIA - TRANSFERISTA') & (pd.to_datetime(df_vendas['Data_Venda']).dt.year>=2025), ['Vendedor', 'Setor']] = ['LETICIA - GUIA', 'Guia']

        df_vendas.loc[(df_vendas['Vendedor']=='LETICIA - TRANSFERISTA') & (pd.to_datetime(df_vendas['Data_Venda']).dt.year<2025), ['Vendedor', 'Setor']] = ['LETICIA - PDV', 'Desks']

        return df_vendas

    def ajustar_pdvs_facebook(df_vendas):

        mask_ref = (df_vendas['Vendedor'].isin(['RAQUEL - PDV', 'VALERIA - PDV', 'ROBERTA - PDV', 'LETICIA - PDV'])) & (pd.to_datetime(df_vendas['Data_Venda']).dt.year<2025) & \
            (df_vendas['Canal_de_Vendas']=='Facebook')
        
        df_vendas.loc[mask_ref, 'Setor'] = 'Guia'

        df_vendas.loc[mask_ref, 'Vendedor'] = df_vendas.loc[mask_ref, 'Vendedor'].apply(lambda x: x.replace('- PDV', '- GUIA'))

        return df_vendas

    def ajustar_colunas_data_venda_mes_ano_total_paxs(df_vendas, coluna_data):

        df_vendas[coluna_data] = pd.to_datetime(df_vendas[coluna_data]).dt.date

        df_vendas = adicionar_colunas_ano_mes(
                        df_vendas, 
                        coluna_data=coluna_data
                    )

        df_vendas['Total Paxs'] = df_vendas['Total_ADT'].fillna(0) + df_vendas['Total_CHD'].fillna(0) / 2

        return df_vendas

    # Puxando as vendas do Phoenix

    st.session_state.df_vendas = gerar_df_phoenix(
        st.session_state.base_luck, 
        '''SELECT * FROM vw_bi_vendas'''
    )

    # Puxando as vendas lançadas manualmente na planilha

    gerar_df_vendas_manuais(
        st.session_state.id_gsheet_bi_vendas, 
        'BD - Vendas Manuais', 
        'df_vendas_manuais', 
        ['Valor_Venda', 'Desconto_Global_Por_Servico', 'Total_ADT', 'Total_CHD'], 
        ['Data_Venda']
    )

    df_vendas = pd.concat(
        [st.session_state.df_vendas, st.session_state.df_vendas_manuais], 
        ignore_index=True
    )

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df_vendas['Setor'] = df_vendas['Setor'].replace(
            'Transferista', 
            'Guia'
        )

        # Ajustando nomes de letícia e soraya pra identificar o setor correto

        df_vendas = ajustar_nomes_leticia_soraya(
            df_vendas
        )

        # Identificando como guia Raquel, Valeria, Roberta e Letícia quando o canal de vendas é Facebook e o ano é antes de 2025

        df_vendas = ajustar_pdvs_facebook(
            df_vendas
        )

    # Ajustando formato de Data_Venda, criando coluna Mes_Ano e criando coluna Total Paxs

    df_vendas = ajustar_colunas_data_venda_mes_ano_total_paxs(
        df_vendas,
        'Data_Venda'
    )

    # Criando coluna setor, identificando pessoal da logistica e colocando a meta p/ cada setor

    df_vendas = criar_coluna_setor_definir_metas(
        df_vendas
    )

    return df_vendas

def transformar_linhas_em_colunas(df_receitas_mensal, coluna_a_transformar, coluna_valor):

    df = df_receitas_mensal[['Ano', 'Mes', 'Mes_Ano']].drop_duplicates().sort_values(by=['Ano', 'Mes']).reset_index(drop=True)

    df_insercao = df_receitas_mensal.groupby(['Mes_Ano', coluna_a_transformar], as_index=False)[coluna_valor].sum()

    for tipo in sorted(df_insercao[coluna_a_transformar].unique()):

        df = df.merge(df_insercao[df_insercao[coluna_a_transformar] == tipo][['Mes_Ano', coluna_valor]], on='Mes_Ano', how='left')

        df.rename(columns={coluna_valor: f'{tipo}'}, inplace=True)

        df[f'{tipo}'] = df[f'{tipo}'].fillna(0)

    return df

def gerar_df_dre_mensal():

    def gerar_df_dre_mensal(df_base):

        if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

            df_dre_mensal = df_base[
                [
                    'Ano', 
                    'Mes', 
                    'Mes_Ano'
                ]
            ].drop_duplicates().sort_values(by=['Ano', 'Mes']).reset_index(drop=True)

        elif st.session_state.base_luck == 'test_phoenix_salvador':

            df_receitas = df_base[
                df_base['status_titulo']=='RECEBIDO'
            ][
                [
                    'Ano', 
                    'Mes', 
                    'Mes_Ano', 
                    'data_vencimento', 
                    'valor_documento', 
                    'descricao', 
                    'descricao_padrao', 
                    'descricaoDRE'
                ]
            ].reset_index(drop=True)

            df_dre_mensal = df_receitas.groupby(
                [
                    'Ano', 
                    'Mes', 
                    'Mes_Ano', 
                    'descricaoDRE'
                ], 
                as_index=False
            )['valor_documento'].sum()

        return df_dre_mensal

    def gerar_df_dre_atualizado(df_dre_mensal):

        df_omie = st.session_state.df_big_query[['Data_venc', 'Desc_Depto', 'Valor_Depto', 'Descricao']].copy()

        df_omie.rename(columns = {'Descricao': 'Categoria'}, inplace=True)

        df_omie['Data_venc'] = pd.to_datetime(df_omie['Data_venc']).dt.date

        df_omie = adicionar_colunas_ano_mes(df_omie, 'Data_venc')

        df_base = st.session_state.df_dre.copy()

        df_base = pd.concat([df_base, df_omie], ignore_index=True)

        df_base['Desc_Depto'] = df_base['Desc_Depto'].str.upper()

        df_base = df_base[df_base['Valor_Depto'] != 0].reset_index(drop=True)

        df_base['Categoria'] = df_base['Categoria'].replace(st.session_state.dict_categorias_alteradas)

        df_base = df_base[~df_base['Categoria'].isin(st.session_state.df_remover_categorias['Categoria'])].reset_index(drop=True)

        df_categ = st.session_state.df_categoria_omie.drop_duplicates(subset=['Categoria_OMIE'])

        df_merged = df_base.merge(
            df_categ[['Categoria_OMIE', 'Tipo']], 
            left_on='Categoria', 
            right_on='Categoria_OMIE', 
            how='left',
        )

        df_merged.drop(columns=['Tipo_x'], inplace=True)

        df_merged.rename(columns={'Tipo_y': 'Tipo'}, inplace=True)

        df_merged = df_merged[df_merged['Mes_Ano']<=df_dre_mensal['Mes_Ano'].max()].reset_index(drop=True)

        return df_merged

    def inserir_coluna_margem_mc(df_1, df_2):

        df_insercao = df_2[df_2['Categoria'].isin(st.session_state.lista_despesas_mc_marcelo)].groupby(['Mes_Ano'], as_index=False)['Valor_Depto'].sum()

        df_1 = df_1.merge(df_insercao, on='Mes_Ano', how='left')

        df_1.rename(columns={'Valor_Depto': 'Despesas MC Marcelo'}, inplace=True)

        df_1['Despesas MC Marcelo'] = df_1['Despesas MC Marcelo'].fillna(0)

        return df_1

    def inserir_despesas(df_dre_mensal):

        df_despesas = st.session_state.df_pagar[
            st.session_state.df_pagar['status_titulo']=='PAGO'
        ][
            [
                'Ano', 
                'Mes', 
                'Mes_Ano', 
                'data_vencimento', 
                'valor_documento', 
                'descricao', 
                'descricao_padrao', 
                'descricaoDRE'
            ]
        ].reset_index(drop=True)

        df_despesas.loc[df_despesas['descricao'].isin(['AUTONOMOS / MEI', 'ADICIONAIS E DIÁRIAS']), 'descricaoDRE'] = 'Custo dos Serviços Prestados'

        df_despesas.loc[df_despesas['descricao'].isin(['OUTRAS DESPESAS INVESTIMENTOS']), 'descricaoDRE'] = 'Ativos'

        df_despesas_mensal = df_despesas.groupby(
            [
                'Ano', 
                'Mes', 
                'Mes_Ano', 
                'descricaoDRE'
            ], 
            as_index=False
        )['valor_documento'].sum()

        st.session_state.df_despesas_mensal = df_despesas.groupby(
            [
                'Ano', 
                'Mes', 
                'Mes_Ano', 
                'descricaoDRE',
                'descricao'
            ], 
            as_index=False
        )['valor_documento'].sum()

        df_despesas_mensal = transformar_linhas_em_colunas(
            df_despesas_mensal, 
            'descricaoDRE', 
            'valor_documento'
        )

        df_dre_mensal = df_dre_mensal.merge(
            df_despesas_mensal, 
            on=['Ano', 'Mes', 'Mes_Ano'], 
            how='left'
        )

        return df_dre_mensal

    def inserir_indices_dre(df):

        if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

            df['Resultado Bruto'] = df['Vendas'] - df['CPV']

            df['Resultado Operacional'] = df['Resultado Bruto'] + df['Receitas Operacionais'] - df['Despesas Operacionais']

            df['Resultado Líquido'] = df['Resultado Operacional'] + df['Receitas Financeiras'] - df['Despesas Impostos'] - df['Investimentos'] - df['Despesas Investimentos'] \
                - df['Despesas Gerenciais']

            df['Despesas Financeiras Totais'] = df['Investimentos'] + df['Despesas Investimentos'] + df['Despesas Gerenciais']
            
            df['Resultado Marcelo'] = df['Vendas'] - df['Despesas MC Marcelo']

            for lista in st.session_state.lista_indices_divisao:

                df[lista[0]] = round(df[lista[1]] / df[lista[2]].replace(0, None), 4)
            
                df[lista[0]] = df[lista[0]].fillna(0)

            df['% Faturamento'] = 1-df['% Opcionais']

        elif st.session_state.base_luck == 'test_phoenix_salvador':

            df.rename(
                columns={
                    'Receita Bruta de Vendas': 'Vendas',
                    'Ativos': 'Investimentos',
                    'Despesas Financeiras': 'Despesas Gerenciais'
                }, 
                inplace=True
            )

            df['Receita Total'] = df['Outras Receitas'] + df['Vendas'] + df['Recuperação de Despesas Variáveis']

            df['CPV'] = df[st.session_state.lista_contas_cpv].sum(axis=1)

            df['Receitas Operacionais'] = df['Outras Receitas'] + df['Recuperação de Despesas Variáveis']

            df['Despesas Operacionais'] = df[st.session_state.lista_contas_despesas_operacionais].sum(axis=1)

            df[['Receitas Financeiras', 'Despesas Investimentos']] = 0

            df['Despesas Impostos'] = df['Impostos'] + df['Outros Tributos']

            df['Resultado Bruto'] = df['Vendas'] - df['CPV']

            df['Resultado Operacional'] = df['Resultado Bruto'] + df['Receitas Operacionais'] - df['Despesas Operacionais']

            df['Resultado Líquido'] = df['Resultado Operacional'] + df['Receitas Financeiras'] - df['Despesas Impostos'] - df['Investimentos'] - df['Despesas Investimentos'] \
                - df['Despesas Gerenciais']

            df['Despesas Financeiras Totais'] = df['Investimentos'] + df['Despesas Investimentos'] + df['Despesas Gerenciais']

            for lista in st.session_state.lista_indices_divisao:

                df[lista[0]] = round(df[lista[1]] / df[lista[2]].replace(0, None), 2)
            
                df[lista[0]] = df[lista[0]].fillna(0)

        return df
    
    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df_dre_mensal = gerar_df_dre_mensal(st.session_state.df_receitas)

        df_dre_atualizado = gerar_df_dre_atualizado(df_dre_mensal)

        st.session_state.df_dre_atualizado = df_dre_atualizado

        df_dre_mensal = transformar_linhas_em_colunas(df_dre_atualizado, 'Tipo', 'Valor_Depto')

        df_dre_mensal = inserir_coluna_margem_mc(df_dre_mensal, df_dre_atualizado)

        df_dre_mensal = df_dre_mensal.merge(st.session_state.df_receitas.drop(columns=['Ano', 'Mes']), on='Mes_Ano', how='left')

    elif st.session_state.base_luck == 'test_phoenix_salvador':
    
        df_dre_mensal = gerar_df_dre_mensal(st.session_state.df_receber)

        df_dre_mensal = transformar_linhas_em_colunas(df_dre_mensal, 'descricaoDRE', 'valor_documento')

        df_dre_mensal = inserir_despesas(df_dre_mensal)

        df_dre_mensal = df_dre_mensal.merge(st.session_state.df_paxs_in_mensal, on='Mes_Ano', how='left')

    df_dre_mensal = inserir_indices_dre(df_dre_mensal)

    return df_dre_mensal

def gerar_df_vendas_agrupado():

    def ajustar_desconto_global(df_vendas):

        if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

            valor_ref = np.where(df_vendas['Data_Venda'] >= datetime.date(2024, 12, 1), 1000, 5000)

            df_vendas['Desconto_Global_Ajustado'] = np.where(
                (df_vendas['Desconto_Global_Por_Servico'].notna()) & (df_vendas['Desconto_Global_Por_Servico'] < valor_ref) & (df_vendas['Servico'] != 'EXTRA'), 
                df_vendas['Desconto_Global_Por_Servico'], 
                0
            )
            
        else:

            df_vendas['Desconto_Global_Ajustado'] = df_vendas['Desconto_Global_Por_Servico']
        
        return df_vendas

    df_vendas = st.session_state.df_vendas_final[pd.notna(st.session_state.df_vendas_final['Setor'])].reset_index(drop=True)

    df_vendas = ajustar_desconto_global(df_vendas)

    df_vendas_agrupado = df_vendas.groupby(['Ano', 'Mes', 'Mes_Ano', 'Setor'], as_index=False).agg({'Valor_Venda': 'sum', 'Valor_Reembolso': 'sum', 'Meta': 'mean'})

    df_vendas_agrupado['Venda_Filtrada'] = df_vendas_agrupado['Valor_Venda'].fillna(0) - df_vendas_agrupado['Valor_Reembolso'].fillna(0)

    df_vendas_agrupado.drop(columns=['Valor_Venda', 'Valor_Reembolso'], inplace=True)

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df_vendas_agrupado = df_vendas_agrupado.merge(st.session_state.df_dre_mensal[['Mes_Ano', 'Paxs']], on='Mes_Ano', how='left')

        df_vendas_agrupado['Ticket_Medio'] = round(df_vendas_agrupado['Venda_Filtrada'] / df_vendas_agrupado['Paxs'].replace(0, None), 2)

        df_vendas_agrupado = df_vendas_agrupado[pd.notna(df_vendas_agrupado['Paxs'])].reset_index(drop=True)

        df_historico = st.session_state.df_historico.merge(
            st.session_state.df_dre_mensal[['Mes_Ano', 'Paxs']], 
            on='Mes_Ano', 
            how='left'
        )

        df_historico.drop(
            columns=['Paxs ADT', 'Paxs CHD'], 
            inplace=True
        )

        df_historico.rename(
            columns={'Valor_Venda': 'Venda_Filtrada'}, 
            inplace=True
        )

        df_historico['Ticket_Medio'] = round(df_historico['Venda_Filtrada'] / df_historico['Paxs'].replace(0, None), 2)

        df_vendas_agrupado = pd.concat(
            [df_vendas_agrupado[df_vendas_agrupado['Mes_Ano'] > pd.Period('2024-04', freq='M')], df_historico[df_historico['Mes_Ano'] <= pd.Period('2024-04', freq='M')]], 
            ignore_index=True
        )

    elif st.session_state.base_luck == 'test_phoenix_salvador':

        df_paxs_in_mensal = st.session_state.df_paxs_in.groupby(['Mes_Ano'], as_index=False)['Total_Paxs'].sum()

        df_vendas_agrupado = df_vendas_agrupado.merge(df_paxs_in_mensal, on='Mes_Ano', how='left')

        df_vendas_agrupado['Ticket_Medio'] = round(df_vendas_agrupado['Venda_Filtrada'] / df_vendas_agrupado['Total_Paxs'].replace(0, None), 2)

    df_vendas_agrupado = df_vendas_agrupado.sort_values(by=['Ano', 'Mes']).reset_index(drop=True)

    return df_vendas_agrupado

def gerar_df_campanha_mensal():

    st.session_state.df_campanha_mensal = st.session_state.df_abastecimentos.groupby(
        ['Ano', 'Mes', 'Mes_Ano', 'Tipo de Veiculo'], 
        as_index=False
    ).agg({
        'Meta do período': 'sum', 
        'Despesa': 'count'
        }
    )

    st.session_state.df_campanha_mensal.rename(
        columns={
            'Tipo de Veiculo': 'Tipo de Veículo', 
            'Meta do período': 'Metas Batidas', 
            'Despesa': 'Quantidade de Abastecimentos'
            }, 
        inplace=True
    )

    st.session_state.df_campanha_mensal['Performance'] = round(st.session_state.df_campanha_mensal['Metas Batidas'] / st.session_state.df_campanha_mensal['Quantidade de Abastecimentos'], 2)

def colher_filtros(row1, df_dre_mensal):

    with row1[0]:

        filtrar_ano = st.multiselect(
            'Excluir Anos', 
            sorted(df_dre_mensal['Ano'].unique())
        )

    with row1[1]:

        filtrar_mes = st.multiselect(
            'Mostrar Apenas Meses',
              sorted(df_dre_mensal['Mes'].unique())
        )

    with row1[2]:

        tipo_analise = st.selectbox(
            'Tipo de Análise', 
            ['Análise de Receitas', 'Análise de Margens'], 
            index=None
        )

    return filtrar_ano, filtrar_mes, tipo_analise

def gerar_df_grafico_vendas_gerais(filtrar_ano, filtrar_mes, df_dre_mensal):

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df_grafico_vendas_gerais = df_dre_mensal[[
            'Ano', 
            'Mes', 
            'Mes_Ano', 
            'Vendas', 
            'Ticket_Medio_Geral', 
            'Paxs', 
            '% Opcionais', 
            '% Faturamento', 
            'Vendas Opcionais', 
            'Ticket_Medio_Opcionais', 
            'Margem Bruta', 
            'Margem Operacional', 
            'Margem Líquida', 
            'Resultado Bruto', 
            'Resultado Operacional', 
            'Resultado Líquido', 
            'CPV', 
            'CPV / Paxs', 
            'Margem Marcelo', 
            'Despesas Operacionais', 
            'Despesas Operacionais / Paxs', 
            'Despesas Financeiras Totais', 
            'Despesas Financeiras Totais / Paxs', 
            'Despesas Impostos', 
            '% Impostos'
            ]
        ].reset_index(drop=True)

    elif st.session_state.base_luck == 'test_phoenix_salvador':

        df_grafico_vendas_gerais = df_dre_mensal[[
            'Ano', 
            'Mes', 
            'Mes_Ano', 
            'Vendas', 
            'Ticket_Medio_Geral', 
            'Total_Paxs',
            'Margem Bruta', 
            'Margem Operacional', 
            'Margem Líquida', 
            'Resultado Bruto', 
            'Resultado Operacional', 
            'Resultado Líquido', 
            'CPV', 
            'CPV / Paxs', 
            'Despesas Operacionais', 
            'Despesas Operacionais / Paxs', 
            'Despesas Financeiras Totais', 
            'Despesas Financeiras Totais / Paxs', 
            'Despesas Impostos', 
            '% Impostos'
            ]
        ].reset_index(drop=True)

        df_grafico_vendas_gerais = df_grafico_vendas_gerais[
            (df_grafico_vendas_gerais['Mes_Ano']<=pd.Timestamp.now().to_period('M')) & 
            (df_grafico_vendas_gerais['Mes_Ano']>=pd.to_datetime('2023-08-01').to_period('M'))
        ]

        df_grafico_vendas_gerais.loc[df_grafico_vendas_gerais['Mes_Ano']==pd.to_datetime('2023-11-01').to_period('M'), 'Ticket_Medio_Geral'] = 0

        df_grafico_vendas_gerais['Total_Paxs'] = df_grafico_vendas_gerais['Total_Paxs'].fillna(0)

    df_grafico_vendas_gerais['Mes_Ano'] = df_grafico_vendas_gerais['Mes_Ano'].dt.strftime('%m/%y')

    df_grafico_vendas_gerais['Vendas'] = df_grafico_vendas_gerais['Vendas'].astype(int)

    if len(filtrar_ano)>0:

        df_grafico_vendas_gerais = df_grafico_vendas_gerais[~df_grafico_vendas_gerais['Ano'].isin(filtrar_ano)].reset_index(drop=True)

    if len(filtrar_mes)>0:

        df_grafico_vendas_gerais = df_grafico_vendas_gerais[df_grafico_vendas_gerais['Mes'].isin(filtrar_mes)].reset_index(drop=True)

    return df_grafico_vendas_gerais

def formatar_moeda(valor, modo=None):

    if modo == 'inteiro':

        moeda_formatada = format_currency(valor, 'BRL', locale='pt_BR')

        return moeda_formatada[:-3]
    
    else:

        return format_currency(valor, 'BRL', locale='pt_BR')

def grafico_barra_linha_RS(df, coluna_vendas, coluna_tm, titulo, nome_eixo_y_2='Ticket Médio'):

    fig = px.bar(
        df, 
        x='Mes_Ano',  
        y=coluna_vendas, 
        title=titulo, 
        labels={coluna_vendas: coluna_vendas}, 
        text=df[coluna_vendas].apply(lambda v: formatar_moeda(v, 'inteiro'))
        )

    fig.update_traces(
        textposition='outside',
        marker=dict(
            color='rgba(173, 216, 230, 0.8)',
            line=dict(color='black', width=1)
            )
        )

    fig.update_layout(
        yaxis_title=coluna_vendas,
        xaxis_title='Mês/Ano',
        yaxis=dict(range=[0, df[coluna_vendas].max()*2]),
        yaxis2=dict(
            title=nome_eixo_y_2, 
            overlaying="y", 
            side="right", 
            showgrid=False, 
            range=[0, df[coluna_tm].max()*1.05]
            )
        )
    
    fig.add_trace(go.Scatter(
        x=df['Mes_Ano'], 
        y=df[coluna_tm], 
        mode='lines+markers+text', 
        name=nome_eixo_y_2, 
        line=dict(width=1, color='black'), 
        marker=dict(size=4), 
        yaxis='y2', 
        line_shape='spline', 
        text=df[coluna_tm].apply(lambda v: formatar_moeda(v, 'inteiro')), 
        textposition='top center', 
        textfont=dict(size=10),
        )
        )

    return fig

def grafico_barra_RS(df, coluna_vendas, titulo):

    fig = px.bar(
        df, 
        x='Mes_Ano',  
        y=coluna_vendas, 
        title=titulo, 
        labels={coluna_vendas: coluna_vendas}, 
        text=df[coluna_vendas].apply(lambda v: formatar_moeda(v, 'inteiro'))
        )

    fig.update_traces(
        textposition='outside',
        marker=dict(
            color='rgba(173, 216, 230, 0.8)',
            line=dict(color='black', width=1)
            )
        )

    fig.update_layout(
        yaxis_title=coluna_vendas,
        xaxis_title='Mês/Ano',
        yaxis=dict(range=[0, df[coluna_vendas].max()*2])
        )

    return fig

def grafico_paxs(df):

    fig = px.line(
        df, 
        x='Mes_Ano',  
        y='Paxs', 
        title='Paxs', 
        labels={'Paxs': 'Paxs'}, 
        text=df['Paxs'].astype(int).astype(str)
        )
    
    fig.update_traces(
        textposition='top center',
        line_shape='spline', 
        line=dict(width=1, color='black'), 
        )
    
    return fig

def plotar_graficos_analise_de_receitas(filtrar_ano, filtrar_mes, df_grafico_vendas_gerais, row2, row3):

    def grafico_linha_opc_vs_fat_percentual(df):

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Mes_Ano'], 
            y=df['% Opcionais'], 
            mode='lines+markers+text', 
            name='% Opcionais', 
            line=dict(width=1, color='black'), 
            marker=dict(size=4), 
            line_shape='spline', 
            text=df['% Opcionais'].apply(lambda v: f'{v*100:.0f}%'), 
            textposition='top center', 
            textfont=dict(size=10),
            )
            )

        fig.add_trace(go.Scatter
            (
            x=df['Mes_Ano'], 
            y=df['% Faturamento'], 
            mode='lines+markers+text', 
            name='% Faturamento', 
            line=dict(width=1, color='rgba(200, 100, 100, 0.8)'), 
            marker=dict(size=4), 
            line_shape='spline', 
            text=df['% Faturamento'].apply(lambda v: f'{v*100:.0f}%'), 
            textposition='top center', 
            textfont=dict(size=10),
            )
            )
        
        fig.update_layout(
            title="% Opcionais vs % Faturamento",
        )
        
        return fig

    def plotar_graficos_vendas_setor(filtrar_ano, filtrar_mes, df_vendas_agrupado, row2):

        def grafico_vendas_setor(df, coluna_vendas, coluna_tm, titulo):

            def formatar_texto(v):

                if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

                    return formatar_moeda(v, 'inteiro')

                elif st.session_state.base_luck == 'test_phoenix_salvador':

                    return formatar_moeda(v)

            fig = px.bar(
                df, 
                x='Mes_Ano',  
                y=coluna_vendas, 
                title=titulo, 
                labels={coluna_vendas: coluna_vendas}, 
                text=df[coluna_vendas].apply(lambda v: formatar_moeda(v, 'inteiro'))
                )

            fig.update_traces(
                textposition='outside',
                marker=dict(
                    color='rgba(173, 216, 230, 0.8)',
                    line=dict(color='black', width=1)
                    )
                )
            
            if df[coluna_tm].max() > df['Meta'].max():

                y2_max = df[coluna_tm].max()*1.05

            else:

                y2_max = df['Meta'].max()*1.05

            fig.update_layout(
                yaxis_title=coluna_vendas,
                xaxis_title='Mês/Ano',
                yaxis=dict(range=[0, df[coluna_vendas].max()*2]),
                yaxis2=dict(
                    title="Ticket Médio", 
                    overlaying="y", 
                    side="right", 
                    showgrid=False, 
                    range=[0, y2_max]
                    )
                )
            
            fig.add_trace(go.Scatter(
                x=df['Mes_Ano'], 
                y=df[coluna_tm], 
                mode='lines+markers+text', 
                name='Ticket Médio', 
                line=dict(
                    width=1, 
                    color='black'
                    ), 
                marker=dict(size=4), 
                yaxis='y2', 
                line_shape='spline', 
                text=df[coluna_tm].apply(formatar_texto), 
                textposition='top center', 
                textfont=dict(size=10),
                )
                )
            
            fig.add_trace(go.Scatter(
                x=df['Mes_Ano'], 
                y=df['Meta'], 
                mode='lines+markers+text', 
                name='Meta T.M.', 
                line=dict(
                    width=1, 
                    color='rgba(200, 100, 100, 0.8)'
                    ), 
                marker=dict(size=4), 
                yaxis='y2', 
                line_shape='spline', 
                text=df['Meta'].apply(formatar_texto), 
                textposition='top center', 
                textfont=dict(size=10),
                )
                )

            return fig

        if len(filtrar_ano)>0:

            df_vendas_agrupado = df_vendas_agrupado[~df_vendas_agrupado['Ano'].isin(filtrar_ano)].reset_index(drop=True)

        if len(filtrar_mes)>0:

            df_vendas_agrupado = df_vendas_agrupado[df_vendas_agrupado['Mes'].isin(filtrar_mes)].reset_index(drop=True)

        i=0

        for setor in df_vendas_agrupado['Setor'].unique():

            df_grafico_vendas_setor = df_vendas_agrupado[(df_vendas_agrupado['Setor'] == setor)].reset_index(drop=True)

            df_grafico_vendas_setor['Mes_Ano'] = df_grafico_vendas_setor['Mes_Ano'].dt.strftime('%m/%y')

            df_grafico_vendas_setor['Venda_Filtrada'] = df_grafico_vendas_setor['Venda_Filtrada'].astype(int)

            fig = grafico_vendas_setor(df_grafico_vendas_setor, 'Venda_Filtrada', 'Ticket_Medio', f'Vendas vs Ticket Médio | {setor}')

            with row2[i%2]:

                st.plotly_chart(fig)

            i+=1

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        with row2[0]:

            fig = grafico_barra_linha_RS(
                df_grafico_vendas_gerais, 
                'Vendas', 
                'Ticket_Medio_Geral', 
                'Vendas vs Ticket Médio | (Opc + Fat)'
            )

            st.plotly_chart(fig)

            fig = grafico_linha_opc_vs_fat_percentual(df_grafico_vendas_gerais)

            st.plotly_chart(fig)

        with row2[1]:

            fig = grafico_paxs(df_grafico_vendas_gerais)

            st.plotly_chart(fig)

            fig = grafico_barra_linha_RS(
                df_grafico_vendas_gerais, 
                'Vendas Opcionais', 
                'Ticket_Medio_Opcionais', 
                'Vendas vs Ticket Médio | (Apenas Opcionais)'
            )

            st.plotly_chart(fig)

    elif st.session_state.base_luck == 'test_phoenix_salvador':

        with row2[0]:

            fig = grafico_barra_linha_RS(
                df_grafico_vendas_gerais, 
                'Vendas', 
                'Ticket_Medio_Geral', 
                'Vendas vs Ticket Médio', 
                nome_eixo_y_2='Ticket Médio'
            )

            st.plotly_chart(fig)

        with row2[1]:

            df_grafico_vendas_gerais.rename(columns={'Total_Paxs': 'Paxs'}, inplace=True)

            df_grafico = df_grafico_vendas_gerais[
                (~df_grafico_vendas_gerais['Mes_Ano'].isin(['08/23', '09/23', '10/23', '11/23']))
            ]

            fig = grafico_paxs(df_grafico)

            st.plotly_chart(fig)

    plotar_graficos_vendas_setor(filtrar_ano, filtrar_mes, st.session_state.df_vendas_agrupado, row3)

def plotar_grafico_resumo_margens(row2, df_grafico_vendas_gerais):

    def grafico_4_linhas_margens(df, titulo):

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Mes_Ano'], 
            y=df['Margem Bruta'], 
            mode='lines+markers+text', 
            name='Margem Bruta', 
            line=dict(width=1, color='black'), 
            marker=dict(size=4), 
            line_shape='spline', 
            text=df['Margem Bruta'].apply(lambda v: f'{v*100:.0f}%'), 
            textposition='top center', 
            textfont=dict(size=10),
            )
            )

        fig.add_trace(go.Scatter
            (
            x=df['Mes_Ano'], 
            y=df['Margem Operacional'], 
            mode='lines+markers+text', 
            name='Margem Operacional', 
            line=dict(width=1, color='rgba(173, 216, 230, 0.8)'), 
            marker=dict(size=4), 
            line_shape='spline', 
            text=df['Margem Operacional'].apply(lambda v: f'{v*100:.0f}%'), 
            textposition='top center', 
            textfont=dict(size=10),
            )
            )
        
        fig.add_trace(go.Scatter
            (
            x=df['Mes_Ano'], 
            y=df['Margem Líquida'], 
            mode='lines+markers+text', 
            name='Margem Líquida', 
            line=dict(width=1, color='rgba(200, 100, 100, 0.8)'), 
            marker=dict(size=4), 
            line_shape='spline', 
            text=df['Margem Líquida'].apply(lambda v: f'{v*100:.0f}%'), 
            textposition='top center', 
            textfont=dict(size=10),
            )
            )
        
        if 'Margem Marcelo' in df.columns:
        
            fig.add_trace(go.Scatter
                (
                x=df['Mes_Ano'],
                y=df['Margem Marcelo'],
                mode='lines+markers+text',
                name='Margem Marcelo',
                line=dict(width=1, color='rgba(100, 200, 100, 0.8)'),
                marker=dict(size=4),
                line_shape='spline',
                text=df['Margem Marcelo'].apply(lambda v: f'{v*100:.0f}%'),
                textposition='top center',
                textfont=dict(size=10),
                )
                )
        
        fig.update_layout(
            title=titulo,
        )
        
        return fig

    with row2[0]:

        fig = grafico_4_linhas_margens(
            df_grafico_vendas_gerais, 
            'Margens'
        )

        st.plotly_chart(fig)

def grafico_resultado_margem(df, coluna_resultado, coluna_margem, titulo):

    fig = px.bar(
        df, 
        x='Mes_Ano',  
        y=coluna_resultado, 
        title=titulo, 
        labels={coluna_resultado: coluna_resultado}, 
        text=df[coluna_resultado].apply(lambda v: formatar_moeda(v, 'inteiro'))
        )

    fig.update_traces(
        textposition='outside',
        marker=dict(
            color='rgba(173, 216, 230, 0.8)',
            line=dict(color='black', width=1)
            )
        )
    
    if df[coluna_resultado].min()<0:

        y_axis_min = df[coluna_resultado].min()*1.1

    else:

        y_axis_min = 0

    if df[coluna_margem].min()<0:

        y_axis_2_min = df[coluna_margem].min()*1.1

    else:

        y_axis_2_min = 0

    fig.update_layout(
        yaxis_title=coluna_resultado,
        xaxis_title='Mês/Ano',
        yaxis=dict(range=[y_axis_min, df[coluna_resultado].max()*2]),
        yaxis2=dict(
            title=coluna_margem, 
            overlaying="y", 
            side="right", 
            showgrid=False, 
            range=[y_axis_2_min, df[coluna_margem].max()*1.1]
            )
        )
    
    fig.add_trace(go.Scatter(
        x=df['Mes_Ano'], 
        y=df[coluna_margem], 
        mode='lines+markers+text', 
        name=coluna_margem, 
        line=dict(width=1, color='black'), 
        marker=dict(size=4), 
        yaxis='y2', 
        line_shape='spline', 
        text=df[coluna_margem].apply(lambda v: f'{v*100:.0f}%'), 
        textposition='top center', 
        textfont=dict(size=10),
        )
        )

    return fig

def plotar_graficos_resultado_bruto(row3, df_grafico_vendas_gerais):

    with row3[0]:

        fig = grafico_resultado_margem(
            df_grafico_vendas_gerais, 
            'Resultado Bruto', 
            'Margem Bruta', 
            'Resultado Bruto vs Margem Bruta'
        )

        st.plotly_chart(fig)

    with row3[1]:

        fig = grafico_barra_linha_RS(
            df_grafico_vendas_gerais, 
            'CPV', 
            'CPV / Paxs', 
            'CPV vs CPV / Paxs', 
            nome_eixo_y_2='CPV / Paxs'
        )

        st.plotly_chart(fig)

def gerar_df_dre_filtrado(filtrar_ano, filtrar_mes):

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df_dre_filtrado = st.session_state.df_dre_atualizado.copy()

    elif st.session_state.base_luck == 'test_phoenix_salvador':
    
        df_dre_filtrado = st.session_state.df_despesas_mensal.copy()

        df_dre_filtrado = df_dre_filtrado.merge(st.session_state.df_paxs_in_mensal, on='Mes_Ano', how='left')

    if len(filtrar_ano)>0:

        df_dre_filtrado = df_dre_filtrado[~df_dre_filtrado['Ano'].isin(filtrar_ano)].reset_index(drop=True)

    if len(filtrar_mes)>0:

        df_dre_filtrado = df_dre_filtrado[df_dre_filtrado['Mes'].isin(filtrar_mes)].reset_index(drop=True)

    return df_dre_filtrado

def gerar_df_categorias(df_dre_filtrado, tipo):

    def inserir_colunas_categorias_despesas(df, df_filtro_tipo):

        df_insercao = df_filtro_tipo.groupby(['Mes_Ano', 'Categoria'], as_index=False)['Valor_Depto'].sum()

        for tipo in sorted(df_insercao['Categoria'].unique()):

            df = df.merge(df_insercao[df_insercao['Categoria'] == tipo][['Mes_Ano', 'Valor_Depto']], on='Mes_Ano', how='left')

            df.rename(columns={'Valor_Depto': f'{tipo}'}, inplace=True)

            df[f'{tipo}'] = df[f'{tipo}'].fillna(0)

        return df

    if st.session_state.base_luck == 'test_phoenix_salvador':

        df_dre_filtrado.rename(
            columns=
                {
                    'descricaoDRE': 'Tipo',
                    'descricao': 'Categoria',
                    'valor_documento': 'Valor_Depto'
                }, 
            inplace=True
        )
        
        if tipo=='CPV':

            df_filtro_tipo = df_dre_filtrado[
                df_dre_filtrado['Tipo'].isin(
                    [
                        'Custo dos Serviços Prestados', 
                        'Deduções de Receita', 
                        'Despesas Variáveis', 
                        'Despesas de Vendas e Marketing'
                    ]
                )
            ].reset_index(drop=True)

        elif tipo=='Despesas Operacionais':

            df_filtro_tipo = df_dre_filtrado[
                df_dre_filtrado['Tipo'].isin(
                    [
                        'Despesas Administrativas', 
                        'Despesas com Pessoal',
                        'Serviços'
                    ]
                )
            ].reset_index(drop=True)

        elif tipo=='Despesas Financeiras':

            df_filtro_tipo = df_dre_filtrado[
                df_dre_filtrado['Tipo'].isin(
                    [
                        'Investimentos', 
                        'Despesas Investimentos', 
                        'Despesas Gerenciais'
                    ]
                )
            ].reset_index(drop=True)

        df_categorias = df_filtro_tipo[['Ano', 'Mes', 'Mes_Ano', 'Total_Paxs']].drop_duplicates().sort_values(by=['Ano', 'Mes']).reset_index(drop=True)

        df_categorias = df_categorias.merge(st.session_state.df_dre_mensal[['Mes_Ano', 'Vendas']], on='Mes_Ano', how='left')

        df_categorias = inserir_colunas_categorias_despesas(
            df_categorias, 
            df_filtro_tipo
        )

        df_categorias = df_categorias.merge(st.session_state.df_metas_kpis, on=['Ano', 'Mes', 'Mes_Ano'], how='left')

        for coluna in st.session_state.df_metas_kpis.drop(columns=['Ano', 'Mes', 'Mes_Ano']).columns:

            df_categorias[coluna] = df_categorias[coluna].fillna(0)

    elif st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        if tipo=='Despesas Financeiras':

            df_filtro_tipo = df_dre_filtrado[df_dre_filtrado['Tipo'].isin(['Investimentos', 'Despesas Investimentos', 'Despesas Gerenciais'])].reset_index(drop=True)

        else:

            df_filtro_tipo = df_dre_filtrado[df_dre_filtrado['Tipo']==tipo].reset_index(drop=True)

        df_categorias = df_filtro_tipo[['Ano', 'Mes', 'Mes_Ano']].drop_duplicates().sort_values(by=['Ano', 'Mes']).reset_index(drop=True)

        df_categorias = df_categorias.merge(
            st.session_state.df_receitas.drop(columns=['Ano', 'Mes']), 
            on='Mes_Ano', 
            how='left'
        )

        df_categorias = df_categorias.merge(
            st.session_state.df_dre_mensal[['Mes_Ano', tipo]], 
            on='Mes_Ano', 
            how='left'
        )

        df_categorias = inserir_colunas_categorias_despesas(
            df_categorias, 
            df_filtro_tipo
        )

    return df_categorias

def inserir_kpis_especificos_cpv(df):

    if st.session_state.base_luck == 'test_phoenix_salvador':

        df['CPV'] = df[
            df.drop(
                columns=
                    [
                        'Ano', 
                        'Mes', 
                        'Mes_Ano'
                    ]
            ).columns
        ].sum(axis=1) 

        df['Autonomo'] = df[
            [
                'GUIA DE PAGAMENTO - ADICIONAL DE SERVIÇO', 
                'GUIA DE PAGAMENTO - GUIAS (OS)', 
                'GUIA DE PAGAMENTO - TERCEIRIZADOS',
                'AUTONOMOS / MEI',
                'ADICIONAIS E DIÁRIAS'
            ]
        ].sum(axis=1)

        df['Comissão Geral'] = df[
            [
                'COMISSÃO PARCEIROS', 
                'GUIA DE PAGAMENTO - COMISSÃO VENDEDOR EXTERNO', 
                'GUIA DE PAGAMENTO DE COMISSÕES VENDEDOR EXTERNO'
            ]
        ].sum(axis=1)

        df.rename(columns={'LOCAÇÃO DE VEICULOS': 'Locação de Veículos'}, inplace=True)

        for coluna in ['Autonomo', 'Locação de Veículos']:

            df[f'% {coluna}'] = round(df[coluna] / df['CPV'], 2)

        regras_divisao = [
            ('Autonomo', 'Total_Paxs', 'Autonomo / Paxs'),
            ('Autonomo', 'Vendas', 'Autonomo / Vendas'),
            ('Locação de Veículos', 'Total_Paxs', 'Locação de Veículos / Paxs'),
            ('Locação de Veículos', 'Vendas', 'Locação de Veículos / Vendas'),
            ('Comissão Geral', 'Vendas', 'Comissão Geral / Vendas')
        ]

        for coluna, divisor, nova_coluna in regras_divisao:

            df[nova_coluna] = round(df[coluna] / df[divisor], 4)

    elif st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df['Comissão Geral'] = df['Comissão de Meta Batida'] + df['Comissão de Parceiros'] + df['Comissão de Vendas'] + df['Comissão de Indicação'] + df['Comissões'] 

        df['Diesel Total'] = df['Diesel Externo'] + df['Diesel Interno']

        for coluna in ['Autonomo', 'Locação de Veículos', 'Manutenção Frota', 'Pneus', 'Diesel Total']:

            df[f'% {coluna}'] = round(df[coluna] / df['CPV'], 2)

        regras_divisao = [
            ('Autonomo', 'Paxs', 'Autonomo / Paxs'),
            ('Autonomo', 'Vendas', 'Autonomo / Vendas'),
            ('Locação de Veículos', 'Paxs', 'Locação de Veículos / Paxs'),
            ('Comissão Geral', 'Vendas', 'Comissão Geral / Vendas'),
            ('Buggys', 'Vendas', 'Buggys / Vendas'),
            ('Locação de Veículos', 'Vendas', 'Locação de Veículos / Vendas'),
            ('Manutenção Frota', 'Km Rodados Total', 'Manutenção Frota / Km'),
            ('Tarifa Cartão', 'Vendas Opcionais', 'Tarifa Cartão / Vendas')
        ]

        for coluna, divisor, nova_coluna in regras_divisao:

            df[nova_coluna] = round(df[coluna] / df[divisor], 4)

    return df

def inserir_kpis_especificos_do(df):

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df['Folha'] = df['13º Salário'] + df['Benefícios'] + df['Encargos Sociais (FGTS, IR, INSS)'] + df['Férias'] + df['Salários'] + df['Vale Alimentação / Refeição'] + df['Vale Transporte']

        df['Folha / Paxs'] = round(df['Folha'] / df['Paxs'], 4)

    elif st.session_state.base_luck == 'test_phoenix_salvador':

        df['Folha'] = df['DESCONTO FOLHA (MULTAS DE TRÂNSITO/AVARIAS, DEMAIS DESCONTOS)'] + df['ESTAGIARIOS'] + df['FGTS'] + df['FÉRIAS'] + df['INSS'] + df['OUTRAS DESPESAS DE PESSOAL'] \
            + df['PARCELAMENTO INSS'] + df['PREMIAÇÃO'] + df['SALÁRIOS/13° SALÁRIO'] + df['VALE ALIMENTAÇÃO/REFEIÇÃO'] + df['VALE TRANSPORTE']
        
        df['Folha / Paxs'] = round(df['Folha'] / df['Total_Paxs'], 4)

    df['Folha / Vendas'] = round(df['Folha'] / df['Vendas'], 4)

    return df

def plotar_gráfico_participacoes_percentuais_cpv(df_cpv_filtrado, row3):

    def grafico_linhas_percentuais(df, lista_colunas, titulo):

        fig = go.Figure()

        for coluna in lista_colunas:

            fig.add_trace(go.Scatter(
                x=df['Mes_Ano'], 
                y=df[coluna], 
                mode='lines+markers+text', 
                name=coluna, 
                line=dict(width=1), 
                marker=dict(size=4), 
                line_shape='spline', 
                text=df[coluna].apply(lambda v: f'{v*100:.0f}%'), 
                textposition='top center', 
                textfont=dict(size=10),
                )
                )
        
        fig.update_layout(
            title=titulo,
        )
        
        return fig
    
    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        df_percentual_categorias = df_cpv_filtrado[
            [
                'Mes_Ano', 
                '% Autonomo', 
                '% Locação de Veículos', 
                '% Manutenção Frota', 
                '% Pneus', 
                '% Diesel Total'
            ]
        ]

        df_percentual_categorias['Mes_Ano'] = df_percentual_categorias['Mes_Ano'].dt.strftime('%m/%y')

        with row3[0]:

            fig = grafico_linhas_percentuais(
                df_percentual_categorias, 
                [
                    '% Autonomo', 
                    '% Locação de Veículos', 
                    '% Manutenção Frota', 
                    '% Pneus', 
                    '% Diesel Total'
                ], 
                '% Despesas / CPV'
            )

            st.plotly_chart(fig)

    elif st.session_state.base_luck == 'test_phoenix_salvador':

        df_percentual_categorias = df_cpv_filtrado[
            [
                'Mes_Ano', 
                '% Autonomo', 
                '% Locação de Veículos'
            ]
        ]

        df_percentual_categorias['Mes_Ano'] = df_percentual_categorias['Mes_Ano'].dt.strftime('%m/%y')

        with row3[0]:

            fig = grafico_linhas_percentuais(
                df_percentual_categorias, 
                [
                    '% Autonomo', 
                    '% Locação de Veículos'
                ], 
                '% Despesas / CPV'
            )

            st.plotly_chart(fig)

def colher_categorias(df, tipo):

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        lista_categorias = list(df[df['Tipo'].isin(tipo)]['Categoria'].unique())

        if 'CPV' in tipo:

            lista_categorias.extend(['Comissão Geral', 'Diesel Total'])

            lista_categorias = sorted(lista_categorias)

            categorias_selecionadas = st.multiselect('Categorias CPV', lista_categorias, default=None)

        elif 'Despesas Operacionais' in tipo:

            lista_categorias.extend(['Folha'])

            lista_categorias = sorted(lista_categorias)

            categorias_selecionadas = st.multiselect('Categorias Despesas Operacionais', sorted(lista_categorias), default=None)

        elif 'Despesas Investimentos' in tipo:

            categorias_selecionadas = st.multiselect('Categorias Despesas Financeiras', sorted(lista_categorias), default=None)

    elif st.session_state.base_luck == 'test_phoenix_salvador':

        if 'CPV' in tipo:

            lista_categorias = list(df[df['Tipo'].isin(st.session_state.lista_contas_cpv)]['Categoria'].unique())

            lista_categorias.extend(['Autonomo', 'Comissão Geral', 'Locação de Veículos'])

            lista_categorias = [item for item in lista_categorias if item != 'LOCAÇÃO DE VEICULOS']

            categorias_selecionadas = st.multiselect('Categorias CPV', sorted(lista_categorias), default=None)

        elif 'Despesas Operacionais' in tipo:

            lista_categorias = list(df[df['Tipo'].isin(st.session_state.lista_contas_despesas_operacionais)]['Categoria'].unique())

            lista_categorias.extend(['Folha'])

            categorias_selecionadas = st.multiselect('Categorias  Despesas Operacionais', sorted(lista_categorias), default=None)

        elif 'Despesas Investimentos' in tipo:

            lista_categorias = list(df[df['Tipo'].isin(st.session_state.lista_contas_despesas_financeiras)]['Categoria'].unique())

            categorias_selecionadas = st.multiselect('Categorias Despesas Financeiras', sorted(lista_categorias), default=None)

    return categorias_selecionadas

def grafico_despesa_meta(df, coluna_despesa, coluna_meta, titulo):

    if df[coluna_despesa].max() < df[coluna_meta].max():

        y_max = df[coluna_meta].max()*1.2

    else:

        y_max = df[coluna_despesa].max()*1.2

    fig = px.bar(
        df, 
        x='Mes_Ano',  
        y=coluna_despesa, 
        title=titulo, 
        labels={coluna_despesa: coluna_despesa}, 
        text=df[coluna_despesa].apply(lambda v: formatar_moeda(v, 'inteiro'))
        )

    fig.update_traces(
        textposition='outside',
        marker=dict(
            color='rgba(173, 216, 230, 0.8)',
            line=dict(color='black', width=1)
            )
        )

    fig.update_layout(
        yaxis_title=coluna_despesa,
        xaxis_title='Mês/Ano',
        yaxis=dict(range=[0, y_max])
        )
    
    fig.add_trace(go.Scatter(
        x=df['Mes_Ano'], 
        y=df[coluna_meta], 
        mode='lines+markers+text', 
        name='Meta', 
        line=dict(width=1, color='rgba(200, 100, 100, 0.8)'), 
        marker=dict(size=4), 
        yaxis='y1', 
        line_shape='spline', 
        text=df[coluna_meta].apply(lambda v: formatar_moeda(v, 'inteiro')), 
        textposition='top center', 
        textfont=dict(size=10),
        )
        )

    return fig

def grafico_despesa(df, coluna_despesa, titulo):

    y_max = df[coluna_despesa].max()*1.2

    fig = px.bar(
        df, 
        x='Mes_Ano',  
        y=coluna_despesa, 
        title=titulo, 
        labels={coluna_despesa: coluna_despesa}, 
        text=df[coluna_despesa].apply(lambda v: formatar_moeda(v, 'inteiro'))
        )

    fig.update_traces(
        textposition='outside',
        marker=dict(
            color='rgba(173, 216, 230, 0.8)',
            line=dict(color='black', width=1)
            )
        )

    fig.update_layout(
        yaxis_title=coluna_despesa,
        xaxis_title='Mês/Ano',
        yaxis=dict(range=[0, y_max])
        )

    return fig

def plotar_gráfico_despesa_meta(categoria, df_grafico, row4, i, df_base):

    if f'Meta {categoria}' in df_base.columns:

        fig = grafico_despesa_meta(df_grafico, categoria, f'Meta {categoria}', f'{categoria} vs Meta {categoria}')

        with row4[i%2]:

            st.plotly_chart(fig)

    else:

        fig = grafico_despesa(df_grafico, categoria, f'{categoria}')

        with row4[i%2]:

            st.plotly_chart(fig)

    i+=1

    return i

def plotar_graficos_kpi_vendas(categoria, df_grafico, row4, i, df_base):

    def grafico_despesa_meta_kpi_vendas(df, coluna_despesa, coluna_meta, titulo):

        if df[coluna_despesa].max() < df[coluna_meta].max():

            y_max = df[coluna_meta].max()*1.2

        else:

            y_max = df[coluna_despesa].max()*1.2

        fig = px.bar(
            df, 
            x='Mes_Ano',  
            y=coluna_despesa, 
            title=titulo, 
            labels={coluna_despesa: coluna_despesa}, 
            text=df[coluna_despesa].apply(lambda v: f'{v*100:.2f}%')
            )

        fig.update_traces(
            textposition='outside',
            marker=dict(
                color='rgba(173, 216, 230, 0.8)',
                line=dict(color='black', width=1)
                )
            )

        fig.update_layout(
            yaxis_title=coluna_despesa,
            xaxis_title='Mês/Ano',
            yaxis=dict(range=[0, y_max]),
            )
        
        fig.add_trace(go.Scatter(
            x=df['Mes_Ano'], 
            y=df[coluna_meta], 
            mode='lines+markers+text', 
            name='Meta', 
            line=dict(width=1, color='rgba(200, 100, 100, 0.8)'), 
            marker=dict(size=4), 
            yaxis='y1', 
            line_shape='spline', 
            text=df[coluna_meta].apply(lambda v: f'{v*100:.2f}%'), 
            textposition='top center', 
            textfont=dict(size=10),
            )
            )

        return fig

    if f'{categoria} / Vendas' in df_base.columns:

        fig = grafico_despesa_meta_kpi_vendas(
            df_grafico, 
            f'{categoria} / Vendas', 
            f'Meta {categoria} / Vendas', 
            f'{categoria} / Vendas vs Meta {categoria} / Vendas'
        )

        with row4[i%2]:

            st.plotly_chart(fig)

        i+=1

    return i

def grafico_1_linha_percentual(df, coluna_y):

    def formatar_text(v):

        if coluna_y!='% Impostos':

            return f'{v*100:.0f}%'
        
        else:

            return f'{v*100:.2f}%'

    fig = px.line(
        df, 
        x='Mes_Ano',  
        y=coluna_y, 
        title=coluna_y, 
        labels={coluna_y: coluna_y}, 
        text=df[coluna_y].apply(formatar_text)
        )
    
    fig.update_traces(
        textposition='top center',
        line_shape='spline', 
        line=dict(width=1, color='black'), 
        )
    
    return fig

def grafico_despesa_meta_kpi_paxs(df, coluna_despesa, coluna_meta, titulo):

    if df[coluna_despesa].max() < df[coluna_meta].max():

        y_max = df[coluna_meta].max()*1.2

    else:

        y_max = df[coluna_despesa].max()*1.2

    fig = px.bar(
        df, 
        x='Mes_Ano',  
        y=coluna_despesa, 
        title=titulo, 
        labels={coluna_despesa: coluna_despesa}, 
        text=df[coluna_despesa].apply(formatar_moeda)
        )

    fig.update_traces(
        textposition='outside',
        marker=dict(
            color='rgba(173, 216, 230, 0.8)',
            line=dict(color='black', width=1)
            )
        )

    fig.update_layout(
        yaxis_title=coluna_despesa,
        xaxis_title='Mês/Ano',
        yaxis=dict(range=[0, y_max]),
        )
    
    fig.add_trace(go.Scatter(
        x=df['Mes_Ano'], 
        y=df[coluna_meta], 
        mode='lines+markers+text', 
        name='Meta', 
        line=dict(width=1, color='rgba(200, 100, 100, 0.8)'), 
        marker=dict(size=4), 
        yaxis='y1', 
        line_shape='spline', 
        text=df[coluna_meta].apply(formatar_moeda), 
        textposition='top center', 
        textfont=dict(size=10),
        )
        )

    return fig

def plotar_graficos_kpi_paxs(categoria, df_grafico, row4, i, df_base):

    if f'{categoria} / Paxs' in df_base.columns:

        fig = grafico_despesa_meta_kpi_paxs(
            df_grafico, 
            f'{categoria} / Paxs', 
            f'Meta {categoria} / Paxs', 
            f'{categoria} / Paxs vs Meta {categoria} / Paxs'
        )

        with row4[i%2]:

            st.plotly_chart(fig)

        i+=1

    return i

def plotar_graficos_kpi_cpv(categoria, df_grafico, row4, i):

    def plotar_graficos_cpv_kpi_paxs(categoria, df_grafico, row4, i):

        if f'{categoria} / Paxs' in st.session_state.df_cpv_filtrado.columns:

            fig = grafico_despesa_meta_kpi_paxs(
                df_grafico, 
                f'{categoria} / Paxs', 
                f'Meta {categoria} / Paxs', 
                f'{categoria} / Paxs vs Meta {categoria} / Paxs'
            )

            with row4[i%2]:

                st.plotly_chart(fig)

            i+=1

        return i
    
    def plotar_graficos_cpv_kpi_km(categoria, df_grafico, row4, i):

        if f'{categoria} / Km' in st.session_state.df_cpv_filtrado.columns:

            fig = grafico_despesa_meta_kpi_paxs(
                df_grafico, 
                f'{categoria} / Km', 
                f'Meta {categoria} / Km', 
                f'{categoria} / Km vs Meta {categoria} / Km'
            )

            with row4[i%2]:

                st.plotly_chart(fig)

            i+=1

        return i
    
    def plotar_graficos_desempenho_combustiveis(categoria, df_grafico, row4, i, lista_tipos_veiculos_nao_diesel, lista_tipos_veiculos_gasolina):

        if 'Diesel' in categoria:

            for tipo_veiculo in st.session_state.df_campanha_mensal['Tipo de Veículo'].unique():

                if not tipo_veiculo in lista_tipos_veiculos_nao_diesel:

                    df_grafico = st.session_state.df_campanha_mensal[st.session_state.df_campanha_mensal['Tipo de Veículo'] == tipo_veiculo].reset_index(drop=True)

                    df_grafico['Mes_Ano'] = df_grafico['Mes_Ano'].dt.strftime('%m/%y')

                    fig = grafico_1_linha_percentual(df_grafico, 'Performance')

                    with row4[i%2]:

                        st.subheader(f'Performance {tipo_veiculo}')

                        st.plotly_chart(fig)

                    i+=1

        elif 'Gasolina' in categoria:

            for tipo_veiculo in lista_tipos_veiculos_gasolina:

                df_grafico = st.session_state.df_campanha_mensal[st.session_state.df_campanha_mensal['Tipo de Veículo'] == tipo_veiculo].reset_index(drop=True)

                df_grafico['Mes_Ano'] = df_grafico['Mes_Ano'].dt.strftime('%m/%y')

                fig = grafico_1_linha_percentual(df_grafico, 'Performance')

                with row4[i%2]:

                    st.subheader(f'Performance SPIN')

                    st.plotly_chart(fig)

                i+=1

        return i

    i = plotar_graficos_cpv_kpi_paxs(
        categoria, 
        df_grafico, 
        row4, 
        i
    )
    
    i = plotar_graficos_kpi_vendas(
        categoria, 
        df_grafico, 
        row4, 
        i,
        st.session_state.df_cpv_filtrado
    )
    
    i = plotar_graficos_cpv_kpi_km(
        categoria, 
        df_grafico, 
        row4, 
        i
    )

    i = plotar_graficos_desempenho_combustiveis(
        categoria, 
        df_grafico, 
        row4, 
        i, 
        ['BUGGY', 'KOMBI', 'SAVEIRO', 'SPIN'], 
        ['SPIN']
    )

    return i

def plotar_graficos_resultado_operacional(row3, df_grafico_vendas_gerais):

    with row3[0]:

        fig = grafico_resultado_margem(
            df_grafico_vendas_gerais, 
            'Resultado Operacional', 
            'Margem Operacional', 
            'Resultado Operacional vs Margem Operacional'
        )

        st.plotly_chart(fig)

    with row3[1]:

        fig = grafico_barra_linha_RS(
            df_grafico_vendas_gerais, 
            'Despesas Operacionais', 
            'Despesas Operacionais / Paxs', 
            'Despesas Operacionais vs Despesas Operacionais / Paxs', 
            nome_eixo_y_2='Despesas Operacionais / Paxs'
        )

        st.plotly_chart(fig)

def plotar_graficos_resultado_liquido(row3, df_grafico_vendas_gerais):

    with row3[0]:

        fig = grafico_resultado_margem(
            df_grafico_vendas_gerais, 
            'Resultado Líquido', 
            'Margem Líquida', 
            'Resultado Líquido vs Margem Líquida'
        )

        st.plotly_chart(fig)

    with row3[1]:

        fig = grafico_barra_linha_RS(
            df_grafico_vendas_gerais, 
            'Despesas Financeiras Totais', 
            'Despesas Financeiras Totais / Paxs', 
            'Despesas Financeiras vs Despesas Financeiras / Paxs', 
            nome_eixo_y_2='Despesas Financeiras Totais / Paxs'
        )

        st.plotly_chart(fig) 

def listar_api(url_api, call_api, titulo_json):

    pagina = 1

    registros_por_pagina = 500 

    todos_registros = []

    while True:

        payload_api = {
            "call": call_api, 
            "app_key": st.session_state.omie_app_key, 
            "app_secret": st.session_state.omie_app_secret,
            "param": [
                {
                    "pagina": pagina, 
                    "registros_por_pagina": registros_por_pagina,
                    "apenas_importado_api": "N"
                }
            ]
        }

        response = requests.post(
            url_api, 
            headers={"Content-Type": "application/json"}, 
            data=json.dumps(payload_api)
        )

        dados = response.json()

        todos_registros.extend(dados.get(titulo_json, []))

        if len(dados.get(titulo_json, [])) < registros_por_pagina:

            break  

        else:

            pagina += 1  

    return todos_registros

def gerar_df_paxs_in():

    request_select = '''SELECT * FROM vw_paxs_in'''
    
    st.session_state.df_paxs_in = gerar_df_phoenix(st.session_state.base_luck, request_select)

    st.session_state.df_paxs_in['Data_Execucao'] = pd.to_datetime(st.session_state.df_paxs_in['Data_Execucao']).dt.date
    
    st.session_state.df_paxs_in['Ano'] = pd.to_datetime(st.session_state.df_paxs_in['Data_Execucao']).dt.year
    
    st.session_state.df_paxs_in['Mes'] = pd.to_datetime(st.session_state.df_paxs_in['Data_Execucao']).dt.month
    
    st.session_state.df_paxs_in['Mes_Ano'] = pd.to_datetime(st.session_state.df_paxs_in['Data_Execucao']).dt.to_period('M')

    st.session_state.df_paxs_in['Total_Paxs'] = st.session_state.df_paxs_in['Total_ADT'].fillna(0) + (st.session_state.df_paxs_in['Total_CHD'].fillna(0) / 2)

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':  

        st.session_state.df_paxs_in = pd.merge(st.session_state.df_paxs_in, st.session_state.df_metas[['Mes_Ano', 'Paxs_Desc']], on='Mes_Ano', how='left')

def ajustar_colunas_dicionarios(df, colunas_dicionarios):

    for coluna in colunas_dicionarios:

        df_expanded = pd.json_normalize(df[coluna])
        df = pd.concat([df, df_expanded], axis=1)

    df = df.drop(columns=colunas_dicionarios)

    return df

def puxar_df_categorias_omie():

    st.session_state.df_categorias = pd.DataFrame(
        listar_api(
            'https://app.omie.com.br/api/v1/geral/categorias/', 
            'ListarCategorias', 
            'categoria_cadastro'
        )
    )

    st.session_state.df_categorias = ajustar_colunas_dicionarios(st.session_state.df_categorias, ['dadosDRE'])

def inserir_categorias_omie(df):

    df = pd.merge(
        df, 
        st.session_state.df_categorias[
            [
                'codigo', 
                'descricao',
                'descricao_padrao', 
                'descricaoDRE'
            ]
        ], 
        left_on='codigo_categoria', 
        right_on='codigo', 
        how='left'
    )

    df = df.drop(
        columns=[
            'codigo_categoria', 
            'codigo'
        ]
    )

    return df

def puxar_df_pagar():

    st.session_state.df_pagar = pd.DataFrame(
        listar_api(
            'https://app.omie.com.br/api/v1/financas/contapagar/', 
            'ListarContasPagar', 
            'conta_pagar_cadastro'
        )
    )

    st.session_state.df_pagar = ajustar_colunas_dicionarios(st.session_state.df_pagar, ['info'])

    st.session_state.df_pagar = inserir_categorias_omie(st.session_state.df_pagar)

    st.session_state.df_pagar = adicionar_colunas_ano_mes(st.session_state.df_pagar, coluna_data='data_vencimento')

def puxar_df_receber():

    st.session_state.df_receber = pd.DataFrame(
        listar_api(
            'https://app.omie.com.br/api/v1/financas/contareceber/', 
            'ListarContasReceber', 
            'conta_receber_cadastro'
        )
    )

    st.session_state.df_receber = ajustar_colunas_dicionarios(st.session_state.df_receber, ['boleto', 'info'])

    st.session_state.df_receber = inserir_categorias_omie(st.session_state.df_receber)

    st.session_state.df_receber = adicionar_colunas_ano_mes(st.session_state.df_receber, coluna_data='data_vencimento')  

st.set_page_config(layout='wide')

if not 'base_luck' in st.session_state:
    
    base_fonte = st.query_params["base_luck"]

    if base_fonte=='jpa':

        st.session_state.base_luck = 'test_phoenix_joao_pessoa'

        st.session_state.id_gsheet_bd_omie_luck = '10s9JA9alUt6-lY093esB7VYluEctlBx7J2kAhPf7XWE'

        st.session_state.id_gsheet_bi_vendas = '1lM3FrBElaVfR-muyt8uFsxDUXOEaSXoPbUlHNJPdgaA'

        st.session_state.id_gsheet_campanha_motoristas = '1Sx6CYMIuFzpeTur5WibxofQfiTk6hyohA3huDCHhKDg'

        st.session_state.lista_despesas_mc_marcelo = [
            'Autonomo', 
            'Locação de Veículos', 
            'Manutenção Frota', 
            'Pneus', 
            'Diesel Interno', 
            'Diesel Externo', 
            'Gasolina'
        ]

        st.session_state.lista_empresas = [
            'Luck', 
            'Mansear', 
            'Kuara'
        ]

        st.session_state.lista_indices_divisao = [
            ['Ticket_Medio_Geral', 'Vendas', 'Paxs'],
            ['Ticket_Medio_Opcionais', 'Vendas Opcionais', 'Paxs'],
            ['% Opcionais', 'Vendas Opcionais', 'Vendas'],
            ['Margem Bruta', 'Resultado Bruto', 'Vendas'],
            ['Margem Operacional', 'Resultado Operacional', 'Vendas'],
            ['Margem Líquida', 'Resultado Líquido', 'Vendas'],
            ['Margem Marcelo', 'Resultado Marcelo', 'Vendas'],
            ['CPV / Paxs', 'CPV', 'Paxs'],
            ['Despesas Operacionais / Paxs', 'Despesas Operacionais', 'Paxs'],
            ['Despesas Financeiras Totais / Paxs', 'Despesas Financeiras Totais', 'Paxs'],
            ['% Impostos', 'Despesas Impostos', 'Vendas'],
        ]

    elif base_fonte=='ssa':

        st.session_state.base_luck = 'test_phoenix_salvador'

        st.session_state.omie_app_key = '3444407555589'

        st.session_state.omie_app_secret = '89e272f43e7dd01b617c6aea7c9a4855'

        st.session_state.id_gsheet_bi_vendas = '1z5cCU_XFdixJqizHY5nKbX6inXBCnUGAAtWEg8rHAr0'

        st.session_state.id_gsheet_dre = '1-9SIWBgfN9IOQWn9wGeJXAF62lLys76E19jJEXPhmuY'

        st.session_state.lista_indices_divisao = [
            ['Ticket_Medio_Geral', 'Vendas', 'Total_Paxs'],
            ['Margem Bruta', 'Resultado Bruto', 'Vendas'],
            ['Margem Operacional', 'Resultado Operacional', 'Vendas'],
            ['Margem Líquida', 'Resultado Líquido', 'Vendas'],
            ['CPV / Paxs', 'CPV', 'Total_Paxs'],
            ['Despesas Operacionais / Paxs', 'Despesas Operacionais', 'Total_Paxs'],
            ['Despesas Financeiras Totais / Paxs', 'Despesas Financeiras Totais', 'Total_Paxs'],
            ['% Impostos', 'Despesas Impostos', 'Vendas'],
        ]

        st.session_state.lista_contas_cpv = [
            'Custo dos Serviços Prestados', 
            'Deduções de Receita', 
            'Despesas Variáveis', 
            'Despesas de Vendas e Marketing'
        ]

        st.session_state.lista_contas_despesas_operacionais = [
            'Despesas Administrativas',
            'Despesas com Pessoal',
            'Serviços'
        ]

        st.session_state.lista_contas_despesas_financeiras = [
            'Ativos',
            'Despesas Financeiras'
        ]

if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

    st.title('DRE | JPA')

    st.divider()

    empresa_selecionada = st.radio(
        'Selecione a empresa', 
        st.session_state.lista_empresas,
        index=None
    )

elif st.session_state.base_luck == 'test_phoenix_salvador':

    st.title('DRE | SSA')

    st.divider()

    empresa_selecionada = 'Luck'

if empresa_selecionada == 'Luck':

    # Puxando dados do Drive e do Phoenix

    if st.session_state.base_luck == 'test_phoenix_joao_pessoa':

        if not 'df_vendas_final' in st.session_state:

            with st.spinner('Carregando dados do Google Drive...'):

                puxar_dados_gsheet(
                    id_gsheet=st.session_state.id_gsheet_bd_omie_luck, 
                    nome_aba='BD_Luck', 
                    nome_df='df_dre', 
                    colunas_numero=['Valor_Depto'], 
                    colunas_data=['Data_venc'], 
                    coluna_data_ano_mes='Data_venc'
                )
                
                puxar_dados_gsheet(
                    id_gsheet=st.session_state.id_gsheet_bd_omie_luck, 
                    nome_aba='BD_Historico_Marcelo', 
                    nome_df='df_receitas'
                )
                
                puxar_dados_gsheet(
                    id_gsheet=st.session_state.id_gsheet_bi_vendas, 
                    nome_aba='BD - Metas', 
                    nome_df='df_metas'
                )

                puxar_df_historico(
                    st.session_state.id_gsheet_bi_vendas, 
                    'BD - Historico', 
                    'df_historico'
                )

                puxar_df_campanha(
                    st.session_state.id_gsheet_campanha_motoristas, 
                    'BD - Historico', 
                    'df_abastecimentos', 
                    ['Média', 'Meta', 'Mes', 'Ano', 'Meta do período'], 
                    ['Data de Abastecimento'], 
                    'BD - Frota | Tipo', 'df_frota', 'Data de Abastecimento'
                )

                puxar_aba_simples(
                    st.session_state.id_gsheet_bd_omie_luck, 
                    'BD_Categoria_De_Para', 
                    'df_categoria_omie'
                )

                gerar_df_big_query_tratado()

                gerar_dict_categorias_alteradas()

                puxar_dados_gsheet(
                    id_gsheet=st.session_state.id_gsheet_bd_omie_luck, 
                    nome_aba='BD_Remover_Categorias', 
                    nome_df='df_remover_categorias',
                    colunas_numero='Nenhuma',
                    colunas_data=None,
                    coluna_data_ano_mes='Nenhuma'
                )

            with st.spinner('Carregando dados do Phoenix...'):

                st.session_state.df_vendas_final = gerar_df_vendas_final()

    elif st.session_state.base_luck == 'test_phoenix_salvador':

        if not 'df_paxs_in' in st.session_state:

            with st.spinner('Carregando dados do OMIE...'):

                puxar_df_categorias_omie()

                puxar_df_pagar()

                puxar_df_receber()

            with st.spinner('Carregando dados do Google Drive...'):

                puxar_dados_gsheet(
                    id_gsheet=st.session_state.id_gsheet_bi_vendas, 
                    nome_aba='BD - Metas', 
                    nome_df='df_metas'
                )

                puxar_aba_simples(
                    st.session_state.id_gsheet_bi_vendas, 
                    'Canal de Vendas - Setor', 
                    'df_canal_de_vendas_setor'
                )

                puxar_dados_gsheet(
                    id_gsheet=st.session_state.id_gsheet_dre, 
                    nome_aba='BD - Metas', 
                    nome_df='df_metas_kpis'
                )

            with st.spinner('Carregando dados do Phoenix...'):

                st.session_state.df_vendas_final = gerar_df_vendas_final()

                gerar_df_paxs_in()

    # Gerando dataframes com dados da DRE, Vendas e Campanha
    
    if (
        st.session_state.base_luck == 'test_phoenix_joao_pessoa' 
        and (
            not 'df_dre_mensal' in st.session_state 
            or not 'df_vendas_agrupado' in st.session_state 
            or not 'df_campanha_mensal' in st.session_state
        )
    ):

        st.session_state.df_dre_mensal = gerar_df_dre_mensal()
        
        st.session_state.df_vendas_agrupado = gerar_df_vendas_agrupado()

        gerar_df_campanha_mensal()

    elif (
        st.session_state.base_luck == 'test_phoenix_salvador' 
        and (
            not 'df_paxs_in_mensal' in st.session_state
            or not 'df_dre_mensal' in st.session_state 
            or not 'df_vendas_agrupado' in st.session_state
        )
    ):
        
        st.session_state.df_paxs_in_mensal = st.session_state.df_paxs_in.groupby(['Mes_Ano'], as_index=False)['Total_Paxs'].sum()

        st.session_state.df_dre_mensal = gerar_df_dre_mensal()

        st.session_state.df_vendas_agrupado = gerar_df_vendas_agrupado()
    
    st.header('Filtros')

    row1 = st.columns(4)

    # Colhendo filtros escolhidos pelo usuário

    filtrar_ano, filtrar_mes, tipo_analise = colher_filtros(
        row1, 
        st.session_state.df_dre_mensal
    )

    # Filtrando colunas que vou usar, ajustando formato de Mes_Ano e filtrando anos e meses escolhidos pelo usuário

    st.session_state.df_grafico_vendas_gerais = gerar_df_grafico_vendas_gerais(
        filtrar_ano, 
        filtrar_mes, 
        st.session_state.df_dre_mensal
    )

    if tipo_analise=='Análise de Receitas':

        st.divider()

        st.header('Análise de Receitas')

        st.subheader('Vendas - Opcionais e Faturamento')

        row2 = st.columns(2)

        st.subheader('Vendas por Setor')

        row3 = st.columns(2)

        plotar_graficos_analise_de_receitas(
            filtrar_ano, 
            filtrar_mes, 
            st.session_state.df_grafico_vendas_gerais, 
            row2, 
            row3
        )

    elif tipo_analise=='Análise de Margens':

        st.divider()

        st.header('Análise de Margens') 

        st.subheader('Margens - Bruta, Operacional e Líquida')

        row2 = st.columns(1)

        st.divider()

        st.subheader('Margem Bruta')

        row3 = st.columns(2)

        row4 = st.columns(2)

        # Plotagem de gráficos de resumo de margens, resultado bruto vs margem bruta, CPV vs CPV / Paxs e % Despesas / CPV

        plotar_grafico_resumo_margens(
            row2, 
            st.session_state.df_grafico_vendas_gerais
        )

        plotar_graficos_resultado_bruto(
            row3, 
            st.session_state.df_grafico_vendas_gerais
        )

        # Gerando df_dre_filtrado e df_cpv_filtrado

        if (
            not 'df_dre_filtrado' in st.session_state 
            or not 'df_cpv_filtrado' in st.session_state 
            or not 'df_do_filtrado' in st.session_state 
            or not 'df_df_filtrado' in st.session_state
        ):

            # Criando dataframe apenas com Anos e Meses selecionados pelo usuário

            st.session_state.df_dre_filtrado = gerar_df_dre_filtrado(
                filtrar_ano, 
                filtrar_mes
            )
            
            # Gerando dataframe somente com categorias de CPV em colunas

            df_cpv_filtrado = gerar_df_categorias(
                st.session_state.df_dre_filtrado, 
                'CPV'
            )

            # Inserir Comissão Geral, Diesel Total e % de ['Autonomo', 'Locação de Veículos', 'Manutenção Frota', 'Pneus', 'Diesel Total'] em relação ao CPV

            st.session_state.df_cpv_filtrado = inserir_kpis_especificos_cpv(df_cpv_filtrado)

            # Gerando dataframe somente com categorias de Despesas Operacionais em colunas

            df_do_filtrado = gerar_df_categorias(
                st.session_state.df_dre_filtrado, 
                'Despesas Operacionais'
            )

            st.session_state.df_do_filtrado = inserir_kpis_especificos_do(df_do_filtrado)

            st.session_state.df_df_filtrado = gerar_df_categorias(
                st.session_state.df_dre_filtrado, 
                'Despesas Financeiras'
            )
        
        # Plotando gráfico de participações percentuais de ['Autonomo', 'Locação de Veículos', 'Manutenção Frota', 'Pneus', 'Diesel Total'] em relação ao CPV

        plotar_gráfico_participacoes_percentuais_cpv(
            st.session_state.df_cpv_filtrado, 
            row3
        )

        # Colhendo categorias de CPV escolhidas pelo usuário

        with row4[0]:

            contas_cpv = colher_categorias(
                st.session_state.df_dre_filtrado, 
                ['CPV']
            )

        # Se o usuário selecionar categorias do CPV, plota os gráficos das categorias selecionadas com seus respectivos kpis e metas

        if len(contas_cpv)>0:

            df_grafico = st.session_state.df_cpv_filtrado.copy()

            df_grafico['Mes_Ano'] = df_grafico['Mes_Ano'].dt.strftime('%m/%y')

            for categoria in contas_cpv:

                row4 = st.columns(2)

                with row4[0]:

                    st.subheader(categoria)

                i=0

                i = plotar_gráfico_despesa_meta(
                    categoria, 
                    df_grafico, 
                    row4, 
                    i,
                    st.session_state.df_cpv_filtrado
                )

                i = plotar_graficos_kpi_cpv(
                    categoria, 
                    df_grafico, 
                    row4, 
                    i
                )

        st.divider()

        st.subheader('Margem Operacional')       

        row3 = st.columns(2)

        # Plotando gráficos de Resultado Operacional, Margem Operacional e Despesas Operacionais / Paxs

        plotar_graficos_resultado_operacional(
            row3, 
            st.session_state.df_grafico_vendas_gerais
        )

        # Colhendo categorias de Despesas Operacionais escolhidas pelo usuário

        row4 = st.columns(2)

        with row4[0]:

            contas_do = colher_categorias(
                st.session_state.df_dre_filtrado, 
                ['Despesas Operacionais']
            )

        # Se o usuário selecionar categorias das Despesas Operacionais, plota os gráficos das categorias selecionadas com seus respectivos kpis e metas

        if len(contas_do)>0:

            df_grafico = st.session_state.df_do_filtrado.copy()

            df_grafico['Mes_Ano'] = df_grafico['Mes_Ano'].dt.strftime('%m/%y')

            for categoria in contas_do:

                row4 = st.columns(2)

                with row4[0]:

                    st.subheader(categoria)

                i=0

                i = plotar_gráfico_despesa_meta(
                    categoria, 
                    df_grafico, 
                    row4, 
                    i,
                    st.session_state.df_do_filtrado
                )

                i = plotar_graficos_kpi_vendas(
                    categoria, 
                    df_grafico, 
                    row4, 
                    i,
                    st.session_state.df_do_filtrado
                )

                i = plotar_graficos_kpi_paxs(
                    categoria, 
                    df_grafico, 
                    row4, 
                    i,
                    st.session_state.df_do_filtrado
                )

        st.divider()

        st.subheader('Margem Líquida') 

        row3 = st.columns(2)   

        # Plotando gráficos de Resultado Líquido, Margem Líquida e Despesas Financeiras / Paxs

        plotar_graficos_resultado_liquido(
            row3, 
            st.session_state.df_grafico_vendas_gerais
        )

        # Colhendo categorias de Despesas Financeiras escolhidas pelo usuário

        row4 = st.columns(2)

        with row4[0]:

            contas_df = colher_categorias(
                st.session_state.df_dre_filtrado, 
                ['Investimentos', 'Despesas Investimentos', 'Despesas Gerenciais']
            )  

        # Se o usuário selecionar categorias das Despesas Financeiras, plota os gráficos das categorias selecionadas com seus respectivos kpis e metas

        if len(contas_df)>0:

            df_grafico = st.session_state.df_df_filtrado.copy()

            df_grafico['Mes_Ano'] = df_grafico['Mes_Ano'].dt.strftime('%m/%y')

            for categoria in contas_df:

                row4 = st.columns(2)

                with row4[0]:

                    st.subheader(categoria)

                i=0

                i = plotar_gráfico_despesa_meta(
                    categoria, 
                    df_grafico, 
                    row4, 
                    i,
                    st.session_state.df_df_filtrado
                )

        st.divider()

        st.subheader('Impostos') 

        # Plotando gráfico de % Impostos 

        fig = grafico_1_linha_percentual(st.session_state.df_grafico_vendas_gerais, '% Impostos')

        st.plotly_chart(fig)
