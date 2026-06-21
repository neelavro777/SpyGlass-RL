# SPY 2020-2025 Daily Dataset Description

This document explains exactly what is inside the `SPY_2020_2025_daily.csv` file, how to read it, and what the financial terms mean in plain English.

---

## 1. What is in the CSV File?

If you open `SPY_2020_2025_daily.csv`, you will notice there are **no columns or labels for "SPY", "Apple", "Tesla", or any other stock names**.

The file contains only 6 columns:
`Date`, `Close`, `High`, `Low`, `Open`, and `Volume`.

### Why are there no stock names?

- This entire CSV file is dedicated to a **single financial asset** called **SPY**.
- Because the file is only tracking this one asset, it does not need a column to specify the name of the asset. Every price in a row represents the value of **SPY** on that specific date.

---

## 2. What is SPY?

- **SPY** is the trading code (or "ticker symbol") for the **SPDR S&P 500 ETF Trust**.
- Think of SPY as a single product that acts like a **basket of stocks**. This basket contains small pieces of the **500 largest companies in the United States** (like Apple, Microsoft, Amazon, Nvidia, etc.).
- Instead of having to track or buy 500 different stocks individually, you can track or buy **SPY** to see how the overall US stock market is performing.
- If the largest US companies do well on average, the price of SPY goes up; if they do poorly, it goes down.

---

## 3. Explaining the CSV Columns

Every row in the CSV represents one trading day. Here is what each column means:

| Column Header in CSV | Meaning in Plain English                                                                                    |
| :------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Date**             | The date the trading took place (Year-Month-Day).                                                           |
| **Open**             | The price of one share of SPY when the market opened in the morning (9:30 AM EST).                          |
| **High**             | The highest price that SPY reached at any point during that day.                                            |
| **Low**              | The lowest price that SPY fell to at any point during that day.                                             |
| **Close**            | The price of SPY when the market closed in the afternoon (4:00 PM EST). This is the final value of the day. |
| **Volume**           | The total number of SPY shares that were bought and sold by traders during that day.                        |

---

## 4. Summary Statistics (2020-2024)

Here is a summary of the prices in the dataset over the 5-year period (1,258 trading days):

- **Average Price (Mean)**: SPY's typical price was around **$404.99**.
- **Lowest Price (Min)**: **$204.94** (during the COVID market crash in March 2020).
- **Highest Price (Max)**: **$597.11** (at the end of December 2024).
- **Average Daily Trading Volume**: Around **81.6 million shares** traded daily.

---

## 5. Timeline of Events in the Dataset

This 5-year period (2020 to 2025) was chosen because it has massive price swings (volatility), which is ideal for testing how smart a trading AI can be:

1. **March 2020 (The COVID Crash)**: Prices plunged to their lowest point ($204.94) and volume spiked as people panicked and sold rapidly.
2. **2021 (The Recovery)**: Prices rose steadily as the economy recovered (often called a **Bull Market**).
3. **2022 (The Inflation Drop)**: Prices fell significantly as inflation rose and interest rates were increased (often called a **Bear Market**).
4. **2023-2024 (The AI Boom)**: Prices rallied to record highs, peaking at $597.11 as technology companies surged.
