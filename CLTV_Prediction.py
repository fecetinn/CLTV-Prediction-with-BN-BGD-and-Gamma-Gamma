"""
BG-NBD ve Gamma-Gamma ile CLTV Tahmini / BG-NBD and Gamma-Gamma CLTV Prediction
===============================================================================

FLO müşteri verilerini kullanarak Customer Lifetime Value (CLTV) tahmini
Customer Lifetime Value (CLTV) prediction using FLO customer data

Autor: fecetinn
Tarih / Date: 2025
"""

import numpy as np
import pandas as pd
from openpyxl.utils.datetime import days_to_time
from tabulate import tabulate
import matplotlib.pyplot as plt
from datetime import *
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
import warnings

# Uyarıları kapat / Disable warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


def check_df_tabulate(dataframe, head=10):
    """
        Generate a comprehensive report of a pandas DataFrame with formatted tables.

        This function provides a detailed analysis of a DataFrame including basic information,
        column statistics, numerical summaries, and sample data displays using tabulate
        for enhanced formatting.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The DataFrame to analyze and report on.
        head : int, optional
            Number of rows to display for head and tail sections (default is 10).

        Returns
        -------
        None
            This function prints the report directly to console and does not return any value.

        Notes
        -----
        The function requires the 'tabulate' library to be installed for proper formatting.
        The report includes:
        - Basic DataFrame information (shape, memory usage)
        - Column details (data types, unique values, missing values)
        - Statistical summaries for numerical columns
        - Sample data from beginning and end of DataFrame
        """

    print("#" * 84)
    print("#" * 27, " " * 5, "DATAFRAME REPORT", " " * 5, "#" * 27)
    print("#" * 84)

    # 1. Temel Bilgiler / Basic Information
    basic_data = [
        ["Number of Rows / Satır Sayısı", dataframe.shape[0]],
        ["Number of Columns / Sütun Sayısı", dataframe.shape[1]],
        ["Total Number of Cells / Toplam Hücre Sayısı", dataframe.shape[0] * dataframe.shape[1]],
        ["Memory (MB) / Bellek (MB)", round(dataframe.memory_usage(deep=True).sum() / 1024 ** 2, 2)]
    ]

    print("\n📊 BASIC INFORMATION / TEMEL BİLGİLER")
    print(tabulate(basic_data, headers=["Metric / Metrik", "Value / Değer"], tablefmt="fancy_grid"))

    # 2. Sütun Bilgileri / Column Information
    column_data = []
    for col in dataframe.columns:
        column_data.append([
            col,
            str(dataframe[col].dtype),
            dataframe[col].nunique(),
            dataframe[col].isnull().sum(),
            f"{round((dataframe[col].isnull().sum() / len(dataframe)) * 100, 2)}%"
        ])

    print("\n📋 COLUMN INFORMATION / SÜTUN BİLGİLERİ")
    print(tabulate(column_data,
                   headers=["Column / Sütun", "Type / Tip", "Unique Values / Benzersiz Değer",
                            "NaN Count / NaN Sayısı", "NaN % / NaN Yüzdesi"],
                   tablefmt="fancy_grid"))

    # 3. İstatistikler (Sayısal sütunlar için) / Statistics (for Numerical columns)
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print("\n📈 STATISTICS OF NUMERICAL COLUMNS / SAYISAL SÜTUNLARIN İSTATİSTİKLERİ")
        stats_data = []
        for col in numeric_cols:
            col_data = dataframe[col]
            stats_data.append([
                col,
                round(col_data.mean(), 2),
                round(col_data.std(), 2),
                col_data.min(),
                round(col_data.quantile(0.01), 2),
                round(col_data.quantile(0.1), 2),
                round(col_data.quantile(0.25), 2),
                round(col_data.median(), 2),
                round(col_data.quantile(0.75), 2),
                round(col_data.quantile(0.9), 2),
                round(col_data.quantile(0.99), 2),
                col_data.max()
            ])

        print(tabulate(stats_data,
                       headers=["Column / Sütun", "Mean / Ort", "Std / Std Sap", "Min / Min", "%1", "%10",
                                "Q1", "Median / Medyan", "Q3", "%90", "%99", "Max / Maks"],
                       tablefmt="fancy_outline"))

    # 4. İlk satırlar / Head
    print(f"\n🔝 FIRST {head} ROWS / İLK {head} SATIR")
    print(tabulate(dataframe.head(head), headers='keys', tablefmt="rounded_outline"))

    # 5. Son satırlar / Tail
    print(f"\n🔚 LAST {head} ROWS / SON {head} SATIR")
    print(tabulate(dataframe.tail(head), headers='keys', tablefmt="rounded_outline"))


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Classify DataFrame columns into categorical, numerical, cardinal, and date categories.

    This function automatically categorizes DataFrame columns based on their data types
    and unique value counts. It identifies categorical columns, numerical columns,
    cardinal (high cardinality categorical) columns, and date/datetime columns using
    specified thresholds.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame whose columns will be classified.
    cat_th : int, optional
        Threshold for determining categorical columns from numerical data.
        Numerical columns with unique values less than this threshold will be
        treated as categorical (default is 10).
    car_th : int, optional
        Threshold for determining cardinal columns from categorical data.
        Categorical columns with unique values greater than this threshold will be
        treated as cardinal (default is 20).

    Returns
    -------
    cat_cols : list
        List of column names identified as categorical columns.
        Includes original categorical columns plus numerical columns with
        low cardinality (< cat_th).
    num_cols : list
        List of column names identified as numerical columns.
        Includes numerical columns that are not classified as categorical.
    cat_but_car : list
        List of column names that are categorical but have high cardinality (> car_th).
        These are typically categorical columns that should be handled differently
        due to their high number of unique values.
    date_cols : list
        List of column names identified as date/datetime columns.
        Includes datetime64, datetime, date, and time data types.

    Notes
    -----
    The function uses the following logic:
    1. Identifies date/datetime columns first
    2. Identifies base categorical columns (category, object, bool types)
    3. Finds numerical columns that behave like categorical (low unique values)
    4. Identifies high cardinality categorical columns
    5. Adjusts classifications to avoid overlaps

    The function prints a summary of the classification results including
    observation count, variable count, and counts for each column type.
    """

    # 1. Önce tarih/datetime sütunlarını tanımla / Identify date/datetime columns first
    date_cols = [col for col in dataframe.columns if
                 dataframe[col].dtype.name.startswith(('datetime', 'date', 'time', 'period')) or
                 str(dataframe[col].dtype) in ['datetime64[ns]', 'datetime64[ns, UTC]', 'timedelta64[ns]']]

    # 2. Veri tiplerine göre temel kategorik sütunları tanımla / Identify base categorical columns based on data types
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    # 3. Kategorik gibi davranan sayısal sütunları bul / Find numerical columns that behave like categorical variables
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    # 4. Yüksek kardinaliteli kategorik sütunları tanımla / Identify categorical columns with high cardinality
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_but_car = [col for col in cat_but_car if col not in date_cols]

    # 5. Temel kategorik sütunları kategorik gibi davranan sayısal sütunlarla birleştir
    # Combine base categorical columns with numerical columns that act as categorical
    cat_cols = cat_cols + num_but_cat

    # 6. Yüksek kardinaliteli kategorik sütunları kategorik listeden çıkar
    # Remove high cardinality categorical columns from the categorical list
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # 7. Sayısal sütunları tanımla (int ve float tipleri) / Identify numerical columns (int and float types)
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]

    # 8. Kategorik olarak sınıflandırılan sütunları sayısal listeden çıkar
    # Remove columns that have been classified as categorical from numerical list
    num_cols = [col for col in num_cols if col not in cat_cols]

    # Sınıflandırma sonuçlarının özet istatistiklerini yazdır / Print summary statistics of the classification results
    print(f"Total Observations / Toplam Gözlem: {dataframe.shape[0]}")
    print(f"Total Variables / Toplam Değişken: {dataframe.shape[1]}")
    print(f'Number of cat_cols / Kategorik sütun sayısı: {len(cat_cols)}')
    print(f'Number of num_cols / Sayısal sütun sayısı: {len(num_cols)}')
    print(f'Number of cat_but_car / Kardinal sütun sayısı: {len(cat_but_car)}')
    print(f'Number of num_but_cat / Kategorik gibi sayısal sütun sayısı: {len(num_but_cat)}')
    print(f'Number of date_cols / Tarih sütunu sayısı: {len(date_cols)}')

    return cat_cols, num_cols, cat_but_car, date_cols


def outlier_threshold(dataframe, variable, q1=0.25, q3=0.75, show_plt=False):
    """
    IQR yöntemi kullanarak aykırı değer eşiklerini hesapla
    Calculate outlier thresholds using IQR method

    Parameters / Parametreler
    ----------
    dataframe : pandas.DataFrame
        Değişkeni içeren DataFrame / The DataFrame containing the variable
    variable : str
        Analiz edilecek sayısal sütun adı / Name of the numerical column to analyze
    q1 : float, optional
        IQR hesaplaması için alt çeyrek (varsayılan 0.25) / Lower quantile for IQR calculation (default is 0.25)
    q3 : float, optional
        IQR hesaplaması için üst çeyrek (varsayılan 0.75) / Upper quantile for IQR calculation (default is 0.75)
    show_plt : bool, optional
        Boxplot görselleştirmesi gösterilsin mi (varsayılan False) / Whether to show boxplot visualization (default is False)

    Returns / Dönüş Değeri
    -------
    tuple
        Alt ve üst aykırı değer eşikleri (tam sayıya yuvarlanmış) / Lower and upper outlier thresholds (rounded to integers)
    """
    quantile_1 = dataframe[variable].quantile(q1)
    quantile_3 = dataframe[variable].quantile(q3)
    interquantile_range = quantile_3 - quantile_1
    upper_limit = quantile_3 + 1.5 * interquantile_range
    lower_limit = quantile_1 - 1.5 * interquantile_range

    if show_plt:
        plt.figure(figsize=(10, 6))
        plt.boxplot(dataframe[variable], patch_artist=True)
        plt.axhline(y=upper_limit, color='red', linestyle='--', linewidth=2,
                    label=f'Upper Limit / Üst Limit: {upper_limit:.2f}')
        plt.axhline(y=lower_limit, color='red', linestyle='--', linewidth=2,
                    label=f'Lower Limit / Alt Limit: {lower_limit:.2f}')
        plt.text(1.1, lower_limit, f'Lower Limit / Alt Limit: {lower_limit:.2f}', fontsize=10)
        plt.text(1.1, upper_limit, f'Upper Limit / Üst Limit: {upper_limit:.2f}', fontsize=10)
        plt.text(1.1, (upper_limit + lower_limit) / 2, f'IQR: {interquantile_range:.2f}',
                 fontsize=10, weight='bold')
        plt.title(f'Boxplot for {variable} / {variable} için Kutu Grafiği')
        plt.ylabel('Values / Değerler')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show(block = True)

    return round(lower_limit), round(upper_limit)


def replace_w_threshold(dataframe, variable, q1=0.25, q3=0.75, replace_upper_lmt=False):
    """
    IQR yöntemi kullanarak aykırı değerleri eşik değerlerle değiştir
    Replace outliers with threshold values using IQR method

    Parameters / Parametreler
    ----------
    dataframe : pandas.DataFrame
        Değişkeni içeren DataFrame (yerinde değiştirilir) / The DataFrame containing the variable (modified in-place)
    variable : str
        İşlenecek sayısal sütun adı / Name of the numerical column to process
    q1 : float, optional
        IQR hesaplaması için alt çeyrek (varsayılan 0.25) / Lower quantile for IQR calculation (default is 0.25)
    q3 : float, optional
        IQR hesaplaması için üst çeyrek (varsayılan 0.75) / Upper quantile for IQR calculation (default is 0.75)
    replace_upper_lmt : bool, optional
        Alt aykırı değerler de değiştirilsin mi (varsayılan False) / Whether to replace lower outliers as well (default is False)

    Returns / Dönüş Değeri
    -------
    None
        DataFrame'i yerinde değiştirir / Modifies the DataFrame in-place
    """
    lower_limit, upper_limit = outlier_threshold(dataframe, variable, q1, q3)

    if replace_upper_lmt:
        # Hem alt hem üst aykırı değerleri değiştir / Replace both lower and upper outliers
        dataframe.loc[(dataframe[variable] < lower_limit), variable] = round(lower_limit)

    # Her zaman üst aykırı değerleri değiştir / Always replace upper outliers
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = round(upper_limit)


def main():
    """
    CLTV analiz fonksiyonu /  CLTV analysis function

    Bu fonksiyon BG-NBD ve Gamma-Gamma modellerini kullanarak CLTV analizini gerçekleştirir
    This function performs CLTV analysis using BG-NBD and Gamma-Gamma models
    """
    print("=" * 80)
    print("=" * 80)
    print(" " * 80)
    print("BG-NBD ve Gamma-Gamma ile CLTV Tahmini / BG-NBD and Gamma-Gamma CLTV Prediction")
    print(" " * 80)
    print("=" * 80)
    print("=" * 80)

    # GÖREV 1: VERİYİ HAZIRLA / TASK 1: PREPARE THE DATA
    print("=" * 50)
    print("\n\n\n1️⃣ GÖREV 1: VERİYİ HAZIRLA / TASK 1: PREPARE THE DATA")
    print("-" * 50)

    # Adım 1: Veriyi oku / Step 1: Read the data
    print("Adım 1: Veri okunuyor... / Step 1: Reading data...")
    print("-" * 50)
    df_backup = pd.read_csv("flo_data_20k.csv")
    df = df_backup.copy()
    print("✅ Veri başarıyla okundu! / Data successfully loaded!")

    # Veriyi incele / Examine the data
    check_df_tabulate(df)

    # Adım 2: Sütun tiplerini analiz et / Step 2: Analyze column types
    print("\n\n\nAdım 2: Sütun tipleri analiz ediliyor... / Step 2: Analyzing column types...")
    print("-" * 50)
    cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(df)

    # Adım 3: Aykırı değerleri baskıla / Step 3: Suppress outliers
    print("\n\n\nAdım 3: Aykırı değerler baskılanıyor... (0.01-0.99)/ Step 3: Suppressing outliers... (0.01-0.99)")
    print("-" * 50)
    outlier_columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
                       "customer_value_total_ever_offline", "customer_value_total_ever_online"]

    for col in outlier_columns:
        if col in df.columns:
            print(f"  {col}: ", end="")
            lower, upper = outlier_threshold(df, col, 0.01, 0.99)
            print(f"{col}: \nAlt limit / Lower limit = {lower}, \nÜst limit / Upper limit = {upper}\n\n")
            replace_w_threshold(df, col, 0.01, 0.99, True)

    # Adım 4: Yeni değişkenler oluştur / Step 4: Create new variables
    print("\n\n\nAdım 4: Yeni değişkenler oluşturuluyor... / Step 4: Creating new variables...")
    print("-" * 50)
    df["order_num_ever_TOTAL"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_ever_TOTAL"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    print(
        "✅ Toplam alışveriş sayısı ve değeri değişkenleri oluşturuldu! / Total purchase count and value variables created!")
    print(df.head(10))


    # Adım 5: Tarih değişkenlerini düzenle / Step 5: Process date variables
    print("\n\n\nAdım 5: Tarih değişkenleri düzenleniyor... / Step 5: Processing date variables...")
    print("-" * 50)
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col])
            print(f"  ✅ {col} datetime formatına çevrildi / converted to datetime format")

    print("\n\n✅ GÖREV 1 TAMAMLANDI! / TASK 1 COMPLETED!")





    print("=" * 50)
    # GÖREV 2: CLTV VERİ YAPISINI OLUŞTUR / TASK 2: CREATE CLTV DATA STRUCTURE
    print("\n\n\n2️⃣ GÖREV 2: CLTV VERİ YAPISINI OLUŞTUR / TASK 2: CREATE CLTV DATA STRUCTURE")
    print("-" * 50)

    # Adım 1: Analiz tarihi belirle / Step 1: Set analysis date
    anaylze_date = max(df["last_order_date"]) + pd.Timedelta(days=2)
    print(f"Analiz tarihi / Analysis date: {anaylze_date}")

    # Adım 2: CLTV dataframe oluştur / Step 2: Create CLTV dataframe
    print("\nCLTV veri yapısı oluşturuluyor... / Creating CLTV data structure...")
    df_cltv = pd.DataFrame()

    # master_id (müşteri kimliği / customer ID)
    df_cltv["customer_id"] = df["master_id"]

    # recency_cltv_weekly (haftalık yenilenme / weekly recency)
    df_cltv["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7

    # T_weekly (haftalık müşteri yaşı / weekly customer age)
    df_cltv["T_weekly"] = (anaylze_date - df["first_order_date"]).dt.days / 7

    # frequency (sıklık / frequency)
    df_cltv["frequency"] = df["order_num_ever_TOTAL"]

    # monetary_cltv_avg (ortalama parasal değer / average monetary value)
    df_cltv["monetary_cltv_avg"] = df["customer_value_ever_TOTAL"] / df["order_num_ever_TOTAL"]

    print(f"✅ CLTV veri yapısı oluşturuldu! Shape: {df_cltv.shape} / CLTV data structure created!")
    print("\nCLTV DataFrame ilk 5 satır / First 5 rows:")
    print(df_cltv.head())

    print("\n\n✅ GÖREV 2 TAMAMLANDI! / TASK 2 COMPLETED!")




    print("=" * 50)
    # GÖREV 3: BG/NBD, GAMMA-GAMMA MODELLERİNİ KUR VE CLTV HESAPLA
    # TASK 3: BUILD BG/NBD, GAMMA-GAMMA MODELS AND CALCULATE CLTV
    print("\n\n\n3️⃣ GÖREV 3: BG/NBD, GAMMA-GAMMA MODELLERİNİ KUR / TASK 3: BUILD BG/NBD, GAMMA-GAMMA MODELS")
    print("-" * 50)

    # Adım 1: BG/NBD modelini fit et / Step 1: Fit BG/NBD model
    print("Adım 1: BG/NBD modeli kuruluyor... / Step 1: Building BG/NBD model...")
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(df_cltv['frequency'],
            df_cltv['recency_cltv_weekly'],
            df_cltv['T_weekly'])

    print("✅ BG/NBD modeli başarıyla kuruldu! / BG/NBD model successfully built!")

    # 3 ay ve 6 ay tahminleri / 3-month and 6-month predictions
    print("\nSatış tahminleri hesaplanıyor... / Calculating sales predictions...")
    df_cltv["exp_sales_3_month"] = bgf.predict(12,  # 12 hafta = 3 ay / 12 weeks = 3 months
                                               df_cltv['frequency'],
                                               df_cltv['recency_cltv_weekly'],
                                               df_cltv['T_weekly'])

    df_cltv["exp_sales_6_month"] = bgf.predict(24,  # 24 hafta = 6 ay / 24 weeks = 6 months
                                               df_cltv['frequency'],
                                               df_cltv['recency_cltv_weekly'],
                                               df_cltv['T_weekly'])

    print(
        f"✅ 3 aylık ortalama beklenen satış / 3-month average expected sales: {df_cltv['exp_sales_3_month'].mean():.2f}")
    print(
        f"✅ 6 aylık ortalama beklenen satış / 6-month average expected sales: {df_cltv['exp_sales_6_month'].mean():.2f}")

    # Model grafiğini çiz / Plot model graph
    print("\nBG/NBD model grafiği çiziliyor... / Drawing BG/NBD model graph...")
    plt.figure(figsize=(12, 8))
    plot_period_transactions(bgf)
    plt.title("BG/NBD Model - Predicted vs Actual Transactions / Tahmin Edilen vs Gerçek İşlemler")
    plt.show(block=True)

    # Adım 2: Gamma-Gamma modelini fit et / Step 2: Fit Gamma-Gamma model
    print("\nAdım 2: Gamma-Gamma modeli kuruluyor... / Step 2: Building Gamma-Gamma model...")
    ggf = GammaGammaFitter(penalizer_coef=0.01)

    ggf.fit(df_cltv['frequency'], df_cltv['monetary_cltv_avg'])

    df_cltv["exp_average_value"] = ggf.conditional_expected_average_profit(df_cltv['frequency'],
                                                                           df_cltv['monetary_cltv_avg'])

    print("✅ Gamma-Gamma modeli başarıyla kuruldu! / Gamma-Gamma model successfully built!")
    print(f"✅ Ortalama beklenen değer / Average expected value: {df_cltv['exp_average_value'].mean():.2f} TL")

    print(df_cltv.head(5))

    # Adım 3: 6 aylık CLTV hesapla / Step 3: Calculate 6-month CLTV
    print("\nAdım 3: 6 aylık CLTV hesaplanıyor... / Step 3: Calculating 6-month CLTV...")
    cltv = ggf.customer_lifetime_value(bgf,
                                       df_cltv['frequency'],
                                       df_cltv['recency_cltv_weekly'],
                                       df_cltv['T_weekly'],
                                       df_cltv['monetary_cltv_avg'],
                                       time=6,  # 6 aylık / 6 months
                                       freq="W",  # T'nin frekans bilgisi / T's frequency info
                                       discount_rate=0.01)

    df_cltv["CLTV_Predicted"] = cltv

    print("✅ CLTV hesaplaması tamamlandı! / CLTV calculation completed!")
    print(f"✅ Ortalama 6 aylık CLTV / Average 6-month CLTV: {df_cltv['CLTV_Predicted'].mean():.2f} TL")
    print(f"✅ Maksimum CLTV / Maximum CLTV: {df_cltv['CLTV_Predicted'].max():.2f} TL")

    print(df_cltv.head(5))

    # En yüksek CLTV'li 20 müşteri / Top 20 customers with highest CLTV
    print("\n🏆 En yüksek CLTV'ye sahip 20 müşteri / Top 20 customers with highest CLTV:")
    top_20 = df_cltv.sort_values(by="CLTV_Predicted", ascending=False).head(20)
    print(top_20[['customer_id', 'CLTV_Predicted', 'frequency', 'monetary_cltv_avg', 'exp_sales_6_month']])

    print("\n✅ GÖREV 3 TAMAMLANDI! / TASK 3 COMPLETED!")

    print("=" * 50)
    # GÖREV 4: CLTV DEĞERİNE GÖRE SEGMENTLERİ OLUŞTUR / TASK 4: CREATE SEGMENTS BASED ON CLTV VALUE
    print("\n\n\n4️⃣ GÖREV 4: CLTV SEGMENTASyONU / TASK 4: CLTV SEGMENTATION")
    print("-" * 50)

    # Adım 1: Segmentlere ayır / Step 1: Divide into segments
    print("Adım 1: CLTV'ye göre segmentlere ayrılıyor... / Step 1: Dividing into segments based on CLTV...")
    df_cltv["SEGMENTS"] = pd.qcut(df_cltv["CLTV_Predicted"],
                                  [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
                                  labels=["F", "E", "D", "C", "B", "A"])

    print("✅ Segmentasyon tamamlandı! / Segmentation completed!")

    # Segment analizi / Segment analysis
    print("\n📊 SEGMENT ANALİZİ / SEGMENT ANALYSIS:")
    segment_summary = df_cltv.groupby("SEGMENTS").agg({
        'customer_id': 'count',
        'CLTV_Predicted': ['mean', 'sum', 'std'],
        'frequency': 'mean',
        'monetary_cltv_avg': 'mean',
        'exp_sales_6_month': 'mean'
    }).round(2)

    print(segment_summary)

    # Segment dağılımı grafiği / Segment distribution chart
    plt.figure(figsize=(15, 10))

    # Subplot 1: Segment sayıları / Segment counts
    plt.subplot(2, 2, 1)
    segment_counts = df_cltv['SEGMENTS'].value_counts().sort_index()
    plt.bar(segment_counts.index, segment_counts.values, color='skyblue', edgecolor='black')
    plt.title('Segment Başına Müşteri Sayısı / Customer Count per Segment')
    plt.xlabel('Segmentler / Segments')
    plt.ylabel('Müşteri Sayısı / Customer Count')


    # Subplot 2: Ortalama CLTV / Average CLTV
    plt.subplot(2, 2, 2)
    avg_cltv = df_cltv.groupby('SEGMENTS')['CLTV_Predicted'].mean().sort_index()
    plt.bar(avg_cltv.index, avg_cltv.values, color='lightgreen', edgecolor='black')
    plt.title('Segment Başına Ortalama CLTV / Average CLTV per Segment')
    plt.xlabel('Segmentler / Segments')
    plt.ylabel('Ortalama CLTV (TL) / Average CLTV (TL)')


    # Subplot 3: CLTV dağılımı / CLTV distribution
    plt.subplot(2, 2, 3)
    plt.hist(df_cltv['CLTV_Predicted'], bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.title('CLTV Dağılımı / CLTV Distribution')
    plt.xlabel('CLTV Değeri / CLTV Value')
    plt.ylabel('Frekans / Frequency')


    # Subplot 4: Segment pie chart
    plt.subplot(2, 2, 4)
    plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Segment Dağılımı (%) / Segment Distribution (%)')

    plt.tight_layout()
    plt.show(block=True)

    # Adım 2: İş önerileri / Step 2: Business recommendations
    print("\n💼 ADIM 2: İŞ ÖNERİLERİ / STEP 2: BUSINESS RECOMMENDATIONS")
    print("-" * 30)

    print("\n🅰️ A SEGMENTİ (Premium Müşteriler - En Üst %10) / A SEGMENT (Premium Customers - Top 10%):")
    print("   💰 Ortalama CLTV / Average CLTV:", f"{avg_cltv['A']:.2f} TL")
    print("   📋 Öneriler / Recommendations:")
    print("     • VIP müşteri programları ve özel etkinlikler / VIP customer programs and exclusive events")
    print("     • Kişiselleştirilmiş premium hizmetler / Personalized premium services")
    print("     • Erken erişim hakları ve özel indirimler / Early access rights and special discounts")
    print("     • Dedicated müşteri temsilcisi / Dedicated customer representative")

    print("\n🅲 C SEGMENTİ (Orta Segment - %25-%50) / C SEGMENT (Mid Segment - 25%-50%):")
    print("   💰 Ortalama CLTV / Average CLTV:", f"{avg_cltv['C']:.2f} TL")
    print("   📋 Öneriler / Recommendations:")
    print("     • Çapraz satış ve yükseltme kampanyaları / Cross-selling and upselling campaigns")
    print("     • Kategori bazlı ürün önerileri / Category-based product recommendations")
    print("     • Sadakat puanı artırma programları / Loyalty point increase programs")
    print("     • Seasonal kampanyalar ve promosyonlar / Seasonal campaigns and promotions")

    # Sonuçları kaydet / Save results
    print("\n💾 SONUÇLAR KAYDEDİLİYOR... / SAVING RESULTS...")
    df_cltv.to_csv("cltv_results.csv", index=False)
    print("✅ Sonuçlar 'cltv_sonuclari.csv' dosyasına kaydedildi! / Results saved to 'cltv_sonuclari.csv'!")

    print("\n✅ GÖREV 4 TAMAMLANDI! / TASK 4 COMPLETED!")

    # ÖZET RAPOR / SUMMARY REPORT
    print("\n" + "=" * 80)
    print("📊 ÖZET RAPOR / SUMMARY REPORT")
    print("=" * 80)

    print(f"📈 Toplam Müşteri Sayısı / Total Customer Count: {len(df_cltv):,}")
    print(f"💰 Toplam 6 Aylık CLTV / Total 6-Month CLTV: {df_cltv['CLTV_Predicted'].sum():,.2f} TL")
    print(f"📊 Ortalama CLTV / Average CLTV: {df_cltv['CLTV_Predicted'].mean():.2f} TL")
    print(f"📊 Medyan CLTV / Median CLTV: {df_cltv['CLTV_Predicted'].median():.2f} TL")

    print(f"\n🏆 En Değerli Segment / Most Valuable Segment: A Segmenti ({segment_counts['A']:,} müşteri / customers)")
    print(
        f"💎 A Segmenti Toplam CLTV / A Segment Total CLTV: {df_cltv[df_cltv['SEGMENTS'] == 'A']['CLTV_Predicted'].sum():,.2f} TL")
    print(f"📈 A Segmenti Ortalama CLTV / A Segment Average CLTV: {avg_cltv['A']:,.2f} TL")

    total_cltv_percentage_A = (df_cltv[df_cltv['SEGMENTS'] == 'A']['CLTV_Predicted'].sum() /
                               df_cltv['CLTV_Predicted'].sum() * 100)
    print(f"💯 A Segmenti Toplam CLTV'nin %{total_cltv_percentage_A:.1f}'ini oluşturuyor")
    print(f"💯 A Segment constitutes {total_cltv_percentage_A:.1f}% of total CLTV")

    print("\n" + "=" * 80)
    print("🎉 BG-NBD VE GAMMA-GAMMA CLTV ANALİZİ TAMAMLANDI!")
    print("🎉 BG-NBD AND GAMMA-GAMMA CLTV ANALYSIS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
