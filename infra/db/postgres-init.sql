-- PostgreSQL schema for the banking domain.
--
-- This script runs once on first startup of the postgres container.
-- It creates the relational tables, indexes, and constraints that the
-- application reads through SQL queries.
--
-- Monetary values stored as BIGINT in cents — never floating-point.
-- Timestamps stored as TIMESTAMPTZ — always timezone-aware in finance.

-- ----------------------------------------------------------------------
-- Customers
-- ----------------------------------------------------------------------
CREATE TABLE customers (
    customer_id     VARCHAR(20) PRIMARY KEY,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    email           VARCHAR(255) NOT NULL UNIQUE,
    enrollment_date DATE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------
-- Accounts
-- ----------------------------------------------------------------------
CREATE TABLE accounts (
    account_id     VARCHAR(20) PRIMARY KEY,
    customer_id    VARCHAR(20) NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
    account_type   VARCHAR(20) NOT NULL CHECK (account_type IN ('checking', 'savings', 'credit_card')),
    balance_cents  BIGINT NOT NULL CHECK (balance_cents >= -10000000),  -- allow negative for credit
    currency       CHAR(3) NOT NULL DEFAULT 'USD',
    opened_date    DATE NOT NULL,
    status         VARCHAR(20) NOT NULL CHECK (status IN ('active', 'frozen', 'closed')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_accounts_customer ON accounts(customer_id);

-- ----------------------------------------------------------------------
-- Transactions
-- ----------------------------------------------------------------------
CREATE TABLE transactions (
    transaction_id  VARCHAR(40) PRIMARY KEY,
    account_id      VARCHAR(20) NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
    timestamp       TIMESTAMPTZ NOT NULL,
    amount_cents    BIGINT NOT NULL,  -- positive = credit, negative = debit
    merchant        VARCHAR(255),
    category        VARCHAR(50) NOT NULL,
    status          VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'posted', 'reversed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for the most common query: transactions for an account, recent first.
CREATE INDEX idx_transactions_account_timestamp ON transactions(account_id, timestamp DESC);

-- Index for category-based queries (e.g., "spending on groceries").
CREATE INDEX idx_transactions_account_category ON transactions(account_id, category);