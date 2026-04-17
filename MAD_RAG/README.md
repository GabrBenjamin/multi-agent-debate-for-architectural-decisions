# MAD_RAG - Multi-Agent Debate with Retrieval-Augmented Generation

## Overview

This module implements the **Multi-Agent Debate (MAD) framework enhanced with Retrieval-Augmented Generation (RAG)** for architectural decision-making evaluation. The RAG component enables agents to retrieve and leverage contextual information from software repositories during the debate process, allowing for more informed and repository-aware architectural decisions.

## Purpose

The MAD_RAG variant explores whether augmenting debate agents with repository-specific knowledge improves their ability to replicate architectural decisions documented in Architecture Decision Records (ADRs). By incorporating RAG, agents can access relevant codebase context, historical patterns, and project-specific constraints that may influence architectural choices.

## Structure

- **Core Components**: Python modules implementing the debate framework, agent logic, and RAG integration
- **Utilities**: Helper modules for LLM interaction, state management, and configuration
- **Data Artifacts**: Databases, CSV files, and extracted datasets containing ADR information and experimental results
- **Execution Scripts**: Tools for running experiments, managing debates, and extracting repository data

## Research Context

This implementation supports research on:
- Retrieval-augmented multi-agent systems
- Context-aware architectural decision-making
- Repository-informed LLM reasoning
- Comparative analysis of debate-based vs. retrieval-based approaches
