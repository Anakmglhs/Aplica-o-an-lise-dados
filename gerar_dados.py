import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def executar_pipeline():
    # PASSO 1: Ingestão, Performance e Limpeza
    df_trans = pd.read_csv('transacoes.csv', encoding='latin1')

    # Imputação da mediana do valor agrupada por estado_cliente
    mediana_por_estado = df_trans.groupby('estado_cliente')['valor'].transform(
        'median'
    )
    df_trans['valor'] = df_trans['valor'].fillna(mediana_por_estado)

    # Criação da coluna estática 'plataforma'
    df_trans['plataforma'] = 'Mobile'

    # Leitura e tratamento do arquivo de cotações (interpolação para NaNs)
    df_cot = pd.read_csv('cotacoes.csv', encoding='utf-8')
    df_cot['cotacao_usd'] = df_cot['cotacao_usd'].interpolate(method='linear')

    
    # PASSO 2: Engenharia de Dados & Alinhamento Temporal
    # Conversão para datetime e fuso horário 'America/Sao_Paulo'
    df_trans['data_transacao'] = pd.to_datetime(df_trans['data_transacao'])
    df_trans['data_transacao'] = df_trans['data_transacao'].dt.tz_localize(
        'America/Sao_Paulo'
    )

    # Criação das colunas dia_semana e mes via acessor .dt
    df_trans['dia_semana'] = df_trans['data_transacao'].dt.day_name()
    df_trans['mes'] = df_trans['data_transacao'].dt.month

    # Remoção de transações duplicadas mantendo a primeira ocorrência
    df_trans = df_trans.drop_duplicates(keep='first').reset_index(drop=True)

    # PASSO 3: Operações Vetorizadas e Filtros Bitwise
    # Filtro vetorizado com operadores bitwise (&, |)
    filtro_bitwise = (
        (df_trans['mes'] == 9)
        & (
            (df_trans['estado_cliente'] == 'SP')
            | (df_trans['estado_cliente'] == 'RJ')
        )
        & (df_trans['valor'] > 5000.0)
    )
    df_setembro_sp_rj_alto = df_trans[filtro_bitwise].copy()

    # PASSO 4: Cruzamento de Dados & Agregação
    # Dicionário de risco e atribuição via .map()
    risco_dict = {
        'C100': 'Baixo',
        'C101': 'Alto',
        'C102': 'Médio',
        'C103': 'Baixo',
        'C104': 'Alto',
    }
    df_trans['nivel_risco'] = df_trans['id_cliente'].map(risco_dict)

    # Cruzamento com a tabela de cotações em USD
    df_trans['data_apenas'] = pd.to_datetime(
        df_trans['data_transacao'].dt.date
    )
    df_cot['data'] = pd.to_datetime(df_cot['data'])
    df_merged = pd.merge(
        df_trans,
        df_cot[['data', 'cotacao_usd']],
        left_on='data_apenas',
        right_on='data',
        how='left',
    )
    df_merged['valor_usd'] = (
        df_merged['valor'] / df_merged['cotacao_usd']
    ).round(2)

    # Tabela dinâmica agrupada por mês e nível de risco com subtotais
    tabela_pivot = pd.pivot_table(
        df_merged,
        index='mes',
        columns='nivel_risco',
        values='valor',
        aggfunc='sum',
        margins=True,
        margins_name='Total',
    )

    # PASSO 5: Detecção Estatística de Outliers (Z-Score)
    def calcular_zscore_por_estado(df):
        media_estado = df.groupby('estado_cliente')['valor'].transform('mean')
        std_estado = df.groupby('estado_cliente')['valor'].transform('std')
        return (df['valor'] - media_estado) / std_estado

    df_merged['z_score'] = calcular_zscore_por_estado(df_merged)

    # Identificação de anomalias potenciais (Z > 2.5)
    df_anomalias = df_merged[df_merged['z_score'] > 2.5].copy()

    # PASSO 6: Visualização Gráfica Orientada a Objetos (Matplotlib)
    df_diario = (
        df_merged.groupby(df_merged['data_transacao'].dt.date)['valor']
        .sum()
        .reset_index()
    )
    df_diario['data_transacao'] = pd.to_datetime(df_diario['data_transacao'])
    df_diario = df_diario.sort_values('data_transacao')

    # Média móvel de 7 dias
    df_diario['media_movel_7d'] = (
        df_diario['valor'].rolling(window=7, min_periods=1).mean()
    )

    # API Orientada a Objetos
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(
        df_diario['data_transacao'],
        df_diario['valor'],
        label='Valor Total Diário (R$)',
        color='#0055d4',
        linewidth=1.8,
        marker='o',
        markersize=3.5,
    )
    ax.plot(
        df_diario['data_transacao'],
        df_diario['media_movel_7d'],
        label='Média Móvel (7 Dias)',
        color='#ff7f0e',
        linewidth=2.5,
        linestyle='--',
    )

    ax.set_ylim(bottom=0)
    ax.set_title(
        'Evolução Diária do Volume Transacionado e Média Móvel (7d)',
        fontsize=13,
        fontweight='bold',
        pad=15,
    )
    ax.set_xlabel('Data', fontsize=10)
    ax.set_ylabel('Valor Total (R$)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True)

    plt.xticks(rotation=35)
    plt.tight_layout()

    plt.savefig('relatorio_transacoes.png', dpi=300)
    plt.show()

    return tabela_pivot, df_anomalias, df_setembro_sp_rj_alto


if __name__ == '__main__':
    tabela_pivot, df_anomalias, df_setembro = executar_pipeline()