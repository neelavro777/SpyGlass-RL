# Research Findings: Why Our AI Failed and How to Fix It

## Initial Run Observations

### 1. The Observation: The AI Did Nothing
When we tested our trained AI on new data from 2024-2025, it failed completely. It chose to "Sell/Stay Out" every single day for 189 days straight. It made zero trades and zero profit.

### 2. The Problem: The "Raw Price" Trap
The AI failed because it was confused by the high prices. During its training in 2020-2023, the SPY stock was priced around $250–$450. When it saw test prices over $500, it didn't know what to do because it had never seen numbers that high before. Since it couldn't understand the new numbers, it panicked and chose the safest option: doing nothing. AI models are very bad at handling numbers outside of what they were trained on.

### 3. The Solution: Two Approaches

We looked at two ways to fix this:

#### Option A: Reward Shaping (Bad Idea)
We could punish the AI with a negative score if it stays out of the market for too long. 
*   **Why it fails:** This forces the AI to make a trade, but it doesn't help the AI understand the market. It's like forcing a blindfolded person to drive. The AI will just make random trades to avoid the punishment, which will lose us money.

#### Option B: Percentage Returns (Good Idea)
Instead of feeding the AI raw prices (like $500), we feed it percentage changes (like +1%).
*   **Why it works:** A +1% jump looks exactly the same to the AI whether the stock is at $50 or $5,000. By converting absolute prices into percentages, the AI can recognize the same upward or downward momentum patterns it learned during training, no matter how high the actual stock price gets.

## Environment Findings: How the AI is Scored
The trading environment acts like a video game where your score updates at the end of every level, not just at the end of the game. The AI gets a "drip-feed" reward every single day based on how the stock moved that specific day. 

### A 7-Day Simulation
Assume Action 1 = Buy/Hold (in the market) and Action 0 = Sell/Stay Out (holding cash).

*   **Day 1 (Price: $100):** AI chooses Action 1 (Buy).
*   **Day 2 (Price: $105):** Stock went up $5. Because AI held overnight, Reward is **+$5**. AI chooses Action 1 (Hold).
*   **Day 3 (Price: $102):** Stock dropped $3. Reward is **-$3**. AI chooses Action 1 (Hold).
*   **Day 4 (Price: $110):** Stock surged $8. Reward is **+$8**. AI chooses Action 0 (Sell).
*   **Day 5 (Price: $90):** Stock crashes. AI is safely holding cash, so it doesn't lose money. Reward is **$0**. AI chooses Action 0 (Stay Out).
*   **Day 6 (Price: $80):** Stock crashes again. Reward is **$0**. AI thinks it's the bottom and chooses Action 1 (Buy).
*   **Day 7 (Price: $85):** Stock bounces $5. Reward is **+$5**.

Notice the first trade (Day 1 to 4) made $10 total profit. Instead of one big reward, the environment gave it out daily (+5, -3, +8 = +10). This daily payout helps the AI learn exactly which days were good and bad.

### Why Our Notebook Showed Zero Profit
During the 189-day test in the notebook, the AI panicked on Day 1 at the high $500+ prices and chose Action 0 (Stay Out). 
*   Because it held cash, Day 2's reward was **$0**. 
*   It panicked again and chose Action 0. Day 3's reward was **$0**.
*   This exact loop repeated for all 189 days.

The total reward was just `$0 + $0 + ... + $0 = $0.000`. It started with 1x its money and ended with 1x its money. It simply held cash for 6 months straight!
