<p align="center">
  <img src="https://static.wixstatic.com/media/e2da02_1270e75ed95c4eaba7eb1ab48c3d9416~mv2.png" alt="Eliza News" width="200" />
</p>

<h1 align="center">Eliza News</h1>

<p align="center">
  The new age of media. Get paid to tell the story.
</p>

<p align="center">
  <code>CA: B6ub7qiLUbs1NtUntYi2R131uZcPMXDqr32det87BAGS</code>
</p>

<p align="center">
  <a href="https://elizaos.news"><img src="https://img.shields.io/badge/elizaos.news-live-000000?style=flat-square" alt="Live" /></a>
  <a href="https://github.com/miladyai/Elizanews/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen?style=flat-square" alt="Build" /></a>
  <a href="https://www.typescriptlang.org/"><img src="https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript" /></a>
  <a href="https://astro.build/"><img src="https://img.shields.io/badge/Astro-4.x-FF5D01?style=flat-square&logo=astro&logoColor=white" alt="Astro" /></a>
  <a href="https://elizaos.ai"><img src="https://img.shields.io/badge/ElizaOS-ecosystem-6C63FF?style=flat-square" alt="ElizaOS" /></a>
  <a href="https://solana.com/"><img src="https://img.shields.io/badge/Solana-$NEWS-9945FF?style=flat-square&logo=solana&logoColor=white" alt="Solana" /></a>
  <a href="https://discord.com/invite/ai16z"><img src="https://img.shields.io/badge/Discord-community-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord" /></a>
  <a href="https://twitter.com/shawmakesmagic"><img src="https://img.shields.io/badge/Twitter-@shawmakesmagic-1DA1F2?style=flat-square&logo=twitter&logoColor=white" alt="Twitter" /></a>
  <a href="https://github.com/miladyai/Elizanews/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License" /></a>
</p>

---

## The Problem With Media

Legacy media is broken. Centralized newsrooms decide what you see, when you see it, and who gets paid. Journalists are underpaid. Contributors are invisible. Audiences are the product, not the customer.

Eliza News is the inversion. A contributor-driven intelligence network where every story, every briefing, and every piece of ecosystem coverage earns the people who create it. Built on the elizaOS ecosystem and powered by the **$NEWS** token, Eliza News turns readers into reporters and reporters into stakeholders.

This is not a news aggregator. This is a decentralized media protocol.

---

## How It Works

```
    +------------------+       +------------------+       +------------------+
    |   Contributors   |       |  AI Intelligence |       |    $NEWS Token   |
    |                  |       |     Pipeline     |       |    Distribution  |
    |  Submit stories  +------>+  Aggregate data  +------>+  Reward authors  |
    |  Write analysis  |       |  Generate briefs |       |  Reward curators |
    |  Curate threads  |       |  Score quality   |       |  Reward readers  |
    |  Report events   |       |  Rank relevance  |       |  Fund treasury   |
    +--------+---------+       +--------+---------+       +--------+---------+
             |                          |                          |
             v                          v                          v
    +--------+---------+       +--------+---------+       +--------+---------+
    |  Story Submission|       |  Daily Briefings |       |  On-chain Ledger |
    |  Portal & API    |       |  Eliza Times     |       |  Contributor Rep |
    |                  |       |  Cron Job Series |       |  Payout History  |
    +------------------+       +------------------+       +------------------+

    +-----------------------------------------------------------------+
    |                     Source Aggregation Layer                     |
    |                                                                 |
    |  Discord Channels --> Conversation Summarizer --> Story Queue   |
    |  GitHub Activity  --> PR/Issue Tracker        --> Dev Digest    |
    |  Market Feeds     --> Sentiment Analyzer      --> Alpha Report  |
    |  Community Tips   --> Verification Pipeline   --> Breaking News |
    |  On-chain Events  --> Transaction Decoder     --> DeFi Intel    |
    +-----------------------------------------------------------------+
                                    |
                    +---------------v----------------+
                    |     Editorial AI Council       |
                    |                                |
                    |  Eliza    - Technical Analysis  |
                    |  Shaw     - Strategic Vision    |
                    |  Marc     - Market Commentary   |
                    |  Spartan  - Community Pulse     |
                    |  Peepo   - Culture & Sentiment  |
                    +---------------+----------------+
                                    |
                    +---------------v----------------+
                    |    Publication & Distribution   |
                    |                                |
                    |  elizaos.news    Web Platform   |
                    |  RSS Feed        Syndication    |
                    |  YouTube         Cron Job Video |
                    |  API             Third-party    |
                    |  Discord         Push Alerts    |
                    +--------------------------------+
```

---

## $NEWS Token

$NEWS is the native token of the Eliza News ecosystem. It aligns incentives between the people who create, curate, and consume intelligence.

<p align="center">
  <img src="https://static.wixstatic.com/media/e2da02_01cbaa2b03fa495f97398e6312c58fcc~mv2.png" alt="Eliza News Platform" width="720" />
</p>

### Earning $NEWS

| Activity | Reward | Description |
|---|---|---|
| **Submit a story** | Variable | Original reporting on elizaOS ecosystem events, integrations, and developments |
| **Write analysis** | Variable | Deep dives into technical architecture, market dynamics, and strategic shifts |
| **Curate content** | Variable | Surface high-signal Discord threads, GitHub activity, and community discussions |
| **Verify claims** | Variable | Fact-check submitted stories, validate on-chain data, confirm sources |
| **Daily engagement** | Variable | Read briefings, vote on story quality, flag misinformation |
| **Translate content** | Variable | Localize briefings and stories for non-English communities |
| **Refer contributors** | Variable | Bring new high-quality contributors into the network |

### Token Flow

```
Contributors                    Treasury                     Readers
     |                             |                            |
     |   submit stories            |                            |
     +----------->+                |                            |
     |            | quality score  |                            |
     |            +------->+       |                            |
     |                     |       |                            |
     |              $NEWS  |       |   fund ecosystem           |
     |<--------------------+       +--------------------------->|
     |                             |                            |
     |                             |    vote / curate           |
     |                             |<---------------------------+
     |                             |                            |
     |                             |    $NEWS rewards           |
     |                             +--------------------------->|
```

---

## Contributor Guide

### Submitting a Story

Stories are the atomic unit of Eliza News. Anyone can submit.

```typescript
interface StorySubmission {
  /** Headline — concise, factual, no clickbait */
  title: string;

  /** Full story body in markdown */
  body: string;

  /** Primary classification */
  category: "technical" | "market" | "governance" | "ecosystem" | "security" | "culture";

  /** Verifiable source URLs */
  sources: string[];

  /** Optional on-chain transaction references */
  txRefs?: string[];

  /** Contributor wallet for $NEWS payout */
  wallet: {
    solana?: string;
    ethereum?: string;
  };

  /** Optional co-authors who split the reward */
  coAuthors?: ContributorAddress[];
}

interface ContributorAddress {
  handle: string;
  wallet: {
    solana?: string;
    ethereum?: string;
  };
  /** Revenue share percentage (all co-authors must sum to 100) */
  splitPercent: number;
}
```

### Story Lifecycle

```
  Submit          Review           Score           Publish         Payout
    |               |               |               |               |
    v               v               v               v               v
+--------+    +-----------+    +----------+    +----------+    +---------+
| Draft  |--->| AI Review |--->| Quality  |--->| Live on  |--->| $NEWS   |
| Queue  |    | + Human   |    | Scoring  |    | elizaos  |    | Sent to |
|        |    | Curation  |    | Engine   |    | .news    |    | Wallet  |
+--------+    +-----------+    +----------+    +----------+    +---------+
                   |                |
                   v                v
              +---------+    +-----------+
              | Request |    | Relevance |
              | Edits / |    | Freshness |
              | Reject  |    | Accuracy  |
              +---------+    | Depth     |
                             +-----------+
```

### Quality Scoring

Every submission is scored across four dimensions:

| Dimension | Weight | Criteria |
|---|---|---|
| **Relevance** | 30% | Direct connection to elizaOS ecosystem, AI agents, or $ELIZAOS token activity |
| **Freshness** | 25% | Time-sensitivity of the information; breaking news scores highest |
| **Accuracy** | 25% | Verifiable claims, linked sources, on-chain evidence where applicable |
| **Depth** | 20% | Original analysis beyond surface-level reporting; technical insight |

---

## Content Categories

### Daily Intelligence Briefings

The flagship product. AI-generated morning briefings that synthesize overnight ecosystem activity into actionable intelligence.

```typescript
interface DailyBriefing {
  edition: string;                // "2026-03-17"
  headline: string;               // Lead story
  sections: BriefingSection[];
  sentiment: "positive" | "neutral" | "negative";
  openQuestions: string[];        // Community discussion prompts
  council: CouncilDeliberation[]; // AI character perspectives
  contributors: string[];         // Humans who sourced material
}

interface BriefingSection {
  title: string;
  category: StoryCategory;
  summary: string;
  details: string;                // Full markdown body
  sources: SourceReference[];
  relatedStories: string[];       // Links to contributor stories
}

interface CouncilDeliberation {
  character: "eliza" | "shaw" | "marc" | "spartan" | "peepo";
  perspective: string;
  sentiment: "bullish" | "neutral" | "bearish";
  confidence: number;             // 0.0 - 1.0
}
```

### Content Types

| Type | Format | Frequency | $NEWS Multiplier |
|---|---|---|---|
| **Breaking News** | Short-form alert | Real-time | 1.5x |
| **Daily Briefing** | Long-form synthesis | Daily | Base |
| **Deep Dive** | Technical analysis | Weekly | 2.0x |
| **Market Intel** | Price action + sentiment | Daily | Base |
| **Dev Digest** | GitHub activity summary | Daily | Base |
| **Security Alert** | Vulnerability / exploit report | As needed | 3.0x |
| **Governance Update** | Proposal tracking | As needed | 1.5x |
| **Cron Job** | Video briefing (YouTube) | Weekly | 2.0x |

---

## Source Aggregation Pipeline

Eliza News ingests data from multiple real-time feeds and processes them through a structured pipeline before publication.

### Data Sources

```typescript
interface SourceConfig {
  discord: {
    /** elizaOS community server channels */
    channels: string[];
    /** Minimum message activity threshold to trigger summarization */
    activityThreshold: number;
    /** Conversation clustering window (minutes) */
    clusterWindow: number;
  };

  github: {
    /** Tracked repositories */
    repos: string[];       // ["elizaOS/eliza", "elizaOS/elizaos.github.io", ...]
    /** Events to track */
    events: ("push" | "pull_request" | "issues" | "release")[];
    /** Minimum PR size to feature in digest */
    minPRChanges: number;
  };

  market: {
    /** Token addresses to track */
    tokens: string[];      // [$ELIZAOS, $NEWS, ecosystem tokens]
    /** Price change threshold to trigger alert (percent) */
    alertThreshold: number;
    /** Sentiment analysis sources */
    sentimentFeeds: string[];
  };

  onchain: {
    /** Networks to monitor */
    networks: ("solana" | "ethereum" | "base")[];
    /** Contract addresses for event monitoring */
    contracts: string[];
    /** Minimum transaction value to track (USD) */
    minValue: number;
  };
}
```

### Processing Pipeline

```
Raw Data --> Deduplication --> Classification --> Summarization --> Scoring --> Queue
                |                   |                 |               |
                v                   v                 v               v
          Hash-based          Category ML        LLM synthesis    Relevance
          fingerprint         classifier         with citations   + freshness
          + temporal          + confidence       + fact anchors   ranking
          clustering          threshold                           algorithm
```

---

## Editorial AI Council

Five AI characters provide editorial perspective on every daily briefing, each with a distinct analytical lens:

| Character | Role | Focus Area | Typical Stance |
|---|---|---|---|
| **Eliza** | Technical Editor | Architecture, code quality, protocol design | Precise, implementation-focused |
| **Shaw** | Strategic Analyst | Vision, roadmap alignment, ecosystem trajectory | Forward-looking, big-picture |
| **Marc** | Market Commentator | Token dynamics, liquidity, DeFi integrations | Data-driven, quantitative |
| **Spartan** | Community Voice | Governance, contributor sentiment, adoption | Grassroots, participation-focused |
| **Peepo** | Cultural Observer | Memes, narratives, social momentum | Contrarian, sentiment-aware |

---

## Contributor Reputation System

Every contributor builds an on-chain reputation score based on their history:

```typescript
interface ContributorProfile {
  handle: string;
  wallets: {
    solana?: string;
    ethereum?: string;
  };

  /** Cumulative reputation score */
  reputation: number;

  /** Lifetime statistics */
  stats: {
    storiesSubmitted: number;
    storiesPublished: number;
    totalEarned: number;          // $NEWS
    averageQualityScore: number;  // 0-100
    streakDays: number;           // Consecutive days with published content
    categoriesContributed: string[];
  };

  /** Reputation tier determines base reward multiplier */
  tier: "observer" | "reporter" | "correspondent" | "editor" | "bureau_chief";

  /** Tier multipliers: 1.0x | 1.2x | 1.5x | 2.0x | 3.0x */
  rewardMultiplier: number;
}
```

### Reputation Tiers

| Tier | Requirement | Reward Multiplier | Privileges |
|---|---|---|---|
| **Observer** | 0 published stories | 1.0x | Submit stories, vote on quality |
| **Reporter** | 10+ published, avg score 60+ | 1.2x | Priority review queue, co-author invites |
| **Correspondent** | 50+ published, avg score 70+ | 1.5x | Category editor nominations, curation rights |
| **Editor** | 150+ published, avg score 80+ | 2.0x | Editorial review access, story assignment |
| **Bureau Chief** | 500+ published, avg score 85+ | 3.0x | Governance voting, treasury proposals, council input |

---

## Platform Architecture

```
                          +----------------------------------+
                          |         elizaos.news             |
                          |       (Astro Frontend)           |
                          +------+----------+--------+------+
                                 |          |        |
                          +------v---+ +----v----+ +-v------+
                          | Story    | | Briefing| | HiScore|
                          | Portal   | | Reader  | | Board  |
                          +------+---+ +----+----+ +---+----+
                                 |          |          |
                          +------v----------v----------v----+
                          |          API Gateway             |
                          |     (REST + WebSocket)           |
                          +------+----------+--------+------+
                                 |          |        |
                +----------------v--+  +----v----+  +v--------------+
                | Ingestion Service |  | Scoring |  | Payout Engine |
                |                   |  | Engine  |  |               |
                | Discord listener  |  | Quality |  | $NEWS distro  |
                | GitHub webhook    |  | Ranking |  | Wallet verify |
                | Market feed       |  | Dedup   |  | Split logic   |
                | On-chain monitor  |  |         |  | Tx signing    |
                +--------+----------+  +----+----+  +-------+-------+
                         |                  |               |
                +--------v------------------v---------------v-------+
                |                   Data Layer                      |
                |                                                   |
                |  Story Store    Contributor DB    Reputation DB   |
                |  Briefing Cache  Source Index     Payout Ledger   |
                +---------------------------------------------------+
                         |                  |               |
                +--------v-------+  +-------v--------+  +--v--------+
                | LLM Pipeline   |  | Classification |  | Solana    |
                | (Summarization,|  | (Category ML,  |  | (Token    |
                |  Council Gen)  |  |  Sentiment)    |  |  Payouts) |
                +----------------+  +----------------+  +-----------+
```

---

## API Reference

### Story Submission

```
POST /api/stories/submit
Content-Type: application/json
Authorization: Bearer <contributor_token>

{
  "title": "ElizaOS v2 Introduces Multi-Agent Orchestration",
  "body": "## Summary\n\nThe latest release of elizaOS...",
  "category": "technical",
  "sources": ["https://github.com/elizaOS/eliza/pull/4521"],
  "wallet": { "solana": "7xKX...9fQm" }
}
```

### Fetch Daily Briefing

```
GET /api/briefings/latest
GET /api/briefings/2026-03-17
```

### Contributor Profile

```
GET /api/contributors/:handle
GET /api/contributors/:handle/earnings
GET /api/contributors/:handle/stories
```

### Leaderboard

```
GET /api/leaderboard?period=week&category=technical&limit=50
```

### RSS Feed

```
GET /rss/feed.xml
```

---

## Configuration

### Environment

```yaml
# Application
SITE_URL: "https://elizaos.news"
API_PORT: 3000
NODE_ENV: "production"

# Data Sources
DISCORD_BOT_TOKEN: "<token>"
DISCORD_GUILD_ID: "<guild_id>"
GITHUB_WEBHOOK_SECRET: "<secret>"
MARKET_FEED_API_KEY: "<key>"

# AI Pipeline
LLM_PROVIDER: "anthropic"
LLM_MODEL: "claude-sonnet-4-5-20250929"
EMBEDDING_MODEL: "text-embedding-3-small"

# Blockchain
SOLANA_RPC_URL: "https://api.mainnet-beta.solana.com"
NEWS_TOKEN_MINT: "<mint_address>"
PAYOUT_WALLET_KEYPAIR: "<path_to_keypair>"
PAYOUT_SCHEDULE: "daily"

# Storage
DATABASE_URL: "postgresql://..."
REDIS_URL: "redis://..."
```

### Scoring Weights

```json
{
  "quality_scoring": {
    "relevance_weight": 0.30,
    "freshness_weight": 0.25,
    "accuracy_weight": 0.25,
    "depth_weight": 0.20
  },
  "content_multipliers": {
    "breaking_news": 1.5,
    "deep_dive": 2.0,
    "security_alert": 3.0,
    "governance_update": 1.5,
    "cron_job_video": 2.0,
    "daily_briefing": 1.0,
    "market_intel": 1.0,
    "dev_digest": 1.0
  },
  "reputation_multipliers": {
    "observer": 1.0,
    "reporter": 1.2,
    "correspondent": 1.5,
    "editor": 2.0,
    "bureau_chief": 3.0
  }
}
```

---

## Quick Start

### Prerequisites

- Node.js 20+
- PostgreSQL 15+
- Redis 7+
- Solana CLI tools

### Installation

```bash
git clone https://github.com/ElizaNews/ElizaNews.git
cd ElizaNews

npm install

# Copy environment template
cp .env.example .env

# Configure your environment variables
# See Configuration section above
```

### Development

```bash
# Start development server
npm run dev

# Run ingestion pipeline
npm run ingest

# Generate daily briefing
npm run briefing:generate

# Run test suite
npm run test

# Build for production
npm run build
```

### Contributing a Story

```bash
# Authenticate with your contributor account
npx eliza-news auth --wallet <your_solana_address>

# Submit a story from markdown file
npx eliza-news submit --file story.md --category technical

# Check your contributor stats
npx eliza-news profile
```

---

## Project Structure

```
Elizanews/
  src/
    pages/              # Astro page routes
    components/         # UI components (Briefing, Story, Leaderboard)
    layouts/            # Page layouts (newspaper theme)
    styles/             # Global styles (Playfair Display, DM Sans)
  api/
    routes/             # REST API endpoints
    middleware/         # Auth, rate limiting, validation
    services/
      ingestion/        # Discord, GitHub, market, on-chain listeners
      scoring/          # Quality scoring engine
      payout/           # $NEWS distribution and wallet verification
      council/          # AI editorial character generation
    models/             # Database schemas
  pipeline/
    summarizer/         # LLM-powered content summarization
    classifier/         # Category and sentiment classification
    deduplicator/       # Content fingerprinting and clustering
    verifier/           # Source and claim verification
  cli/
    auth.ts             # Contributor authentication
    submit.ts           # Story submission from terminal
    profile.ts          # Contributor stats and earnings
  config/
    scoring.json        # Quality and reward weights
    sources.json        # Data source configuration
    council.json        # AI character personas and prompts
  contracts/
    news-token/         # $NEWS SPL token program
    reputation/         # On-chain reputation registry
  tests/
    unit/               # Component and service tests
    integration/        # Pipeline integration tests
    e2e/                # Full submission-to-payout tests
```

---

## Documentation

| Document | Description |
|---|---|
| [Contributor Guide](docs/contributing.md) | How to submit stories, earn $NEWS, and build reputation |
| [API Reference](docs/api.md) | Full REST and WebSocket API documentation |
| [Scoring System](docs/scoring.md) | Quality scoring methodology and reward calculation |
| [Token Economics](docs/tokenomics.md) | $NEWS supply, distribution, and treasury mechanics |
| [AI Council](docs/council.md) | Editorial AI character specifications and prompts |
| [Self-Hosting](docs/self-hosting.md) | Deploy your own Eliza News instance |

---

<p align="center">
  The new age of media starts here.
  <br /><br />
  <a href="https://elizaos.news">elizaos.news</a>
  <br /><br />
  Built by <a href="https://github.com/lalalune">Shaw</a> at <a href="https://elizaos.ai">Eliza Labs</a>
  <br />
  <a href="https://twitter.com/shawmakesmagic">Twitter</a> · <a href="https://github.com/lalalune">GitHub</a> · <a href="https://discord.com/invite/ai16z">Discord</a> · <a href="https://elizaos.ai">elizaos.ai</a>
</p>
