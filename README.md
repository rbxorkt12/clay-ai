# Clay AI by SuperDisco

A production-ready, cloud-native multi-agent framework for building intelligent applications, powered by Vercel and E2B sandboxes.

## 🌟 Key Features

* **Cloud-Native**: Deployed on Vercel for instant access - no local setup required
* **E2B Integration**: Secure sandboxed environments for agent code execution
* **Provider Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, Groq)
* **Type-Safe**: Built with Pydantic V2 for robust model validation
* **Production Ready**: Enterprise-grade monitoring, logging, and security
* **CLI Automation**: Built-in CLI preferences for seamless automation
* **Spec-Driven**: Structured approach to agent specifications and behaviors

## 🚀 Quick Start

Unlike traditional frameworks that require local setup, Clay AI is accessible instantly through our Vercel deployment:

```bash
# No installation needed! Access directly at:
https://clay-ai.superdisco.io

# Or use our CLI tool for local development
npm install -g @superdisco/clay-cli
clay init my-project
```

## 🏗️ Architecture

Clay AI is built on a modular architecture with these key components:

```
clay_ai/
├── core/                 # Core framework functionality
│   ├── config.py        # Configuration management
│   ├── sandbox.py       # E2B sandbox integration
│   └── security.py      # Authentication & authorization
│
├── agents/              # Agent implementations
│   ├── base.py         # Base agent class
│   ├── executor.py     # Task execution agent
│   └── coordinator.py  # Multi-agent coordination
│
├── cli/                 # CLI automation tools
│   ├── preferences/    # Customizable CLI settings
│   └── automation/     # Automation scripts
│
└── vercel/             # Vercel deployment configuration
    └── api/            # Serverless API endpoints
```

## 🛠️ Components

### Agents
Autonomous entities that understand, plan, and execute tasks using natural language.

### Tools
Pre-built functions that agents can use to interact with external systems.

### CLI Preferences
Customizable automation settings for different development workflows.

### Spec Stories
Structured specifications for defining agent behaviors and interactions.

## 🌐 Deployment

Clay AI is deployed on Vercel with E2B sandbox integration:

1. **Instant Access**: No local setup required
2. **Secure Execution**: All agent code runs in isolated E2B sandboxes
3. **Scalable**: Automatically scales based on demand
4. **Low Latency**: Edge deployment for optimal performance

## 💻 Development

For local development:

```bash
# Clone the repository
git clone https://github.com/superdisco/clay-ai.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Run locally
python -m clay_ai.server
```

## 📚 Documentation

Full documentation is available at [https://clay-ai.superdisco.io/docs](https://clay-ai.superdisco.io/docs)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with ❤️ by SuperDisco 