# Sandbox Template Guidelines

This document outlines the requirements and best practices for creating sandbox templates in the E2B Code Interpreter environment.

## Template Structure

### Required Files
```
sandbox-templates/[Template Name]/
├── e2b.Dockerfile        # Container configuration
├── e2b.toml             # Template configuration
├── preview.md           # Natural language preview
├── assets/             # Template-specific assets
└── utils/              # Helper functions and utilities
```

## Configuration Files

### e2b.toml
```toml
# Template configuration
name = "Template Name"
version = "0.1.0"
description = "Clear, concise description"

# Startup command
start_cmd = "your startup command"

# Port configuration
[[ports]]
port = 8888  # Example port
description = "Service description"

# Environment setup
[env]
KEY = "value"

# Preview configuration
[preview]
file = "preview.md"

# Template metadata
[metadata]
technologies = [
    "technology-1",
    "technology-2"
]
```

### e2b.Dockerfile
```dockerfile
FROM base-image:version

# Install dependencies
RUN apt-get update && apt-get install -y \
    package1 \
    package2

# Copy template files
COPY . /home/user/

# Set working directory
WORKDIR /home/user
```

## Preview File (preview.md)

The preview.md should follow this structure:
1. Title with environment name
2. Brief introduction
3. Categorized tools/features with emojis
4. Clear usage instructions
5. Port information

Example:
```markdown
# Environment Name

Brief description of the environment.

✨ **Category 1**
- Tool 1 description
- Tool 2 description

🔧 **Category 2**
- Feature 1 details
- Feature 2 details

Access instructions and port information.
```

## Best Practices

### 1. Template Organization
- Use clear, descriptive names for directories and files
- Keep utils/ directory for reusable functions
- Place assets in assets/ directory
- Include README.md for template-specific documentation

### 2. Docker Configuration
- Use official base images
- Minimize layer count
- Include only necessary dependencies
- Set appropriate permissions
- Configure default working directory

### 3. Environment Configuration
- Define all required environment variables
- Document port configurations
- Include clear startup commands
- Set appropriate default values

### 4. Preview Content
- Write clear, concise descriptions
- Use emojis for visual categorization
- List all major features and tools
- Include usage examples
- Specify port numbers and access methods

### 5. Testing
Before submitting a template:
1. Test full installation process
2. Verify all tools are accessible
3. Check preview display
4. Validate port configurations
5. Test startup command
6. Verify environment variables

## Template Categories

Common template categories include:
- Data Analysis (Python, R, Julia)
- Web Development (Node.js, Python, PHP)
- Machine Learning (PyTorch, TensorFlow)
- DevOps (Docker, Kubernetes)
- Database (SQL, NoSQL)

## Port Configuration

Common port assignments:
- 8888: Jupyter Notebook
- 3000: Node.js applications
- 8000: Python web servers
- 5000: Flask applications
- 8501: Streamlit applications

## Security Considerations

1. **Environment Variables**
   - Never commit sensitive values
   - Use .env.template for required variables
   - Document security requirements

2. **Permissions**
   - Set appropriate file permissions
   - Use non-root users when possible
   - Restrict network access appropriately

3. **Dependencies**
   - Use specific versions
   - Regular security updates
   - Minimize attack surface

## Submission Checklist

Before submitting a new template:

- [ ] All required files present
- [ ] e2b.toml properly configured
- [ ] e2b.Dockerfile optimized
- [ ] preview.md clear and informative
- [ ] Ports correctly configured
- [ ] Dependencies documented
- [ ] Security considerations addressed
- [ ] Testing completed
- [ ] Documentation updated

## Example Templates

Reference implementations:
1. DuckDB Analyst
2. Next.js Developer
3. Streamlit Developer
4. Vue.js Developer
5. Gradio Developer

## Maintenance

Regular maintenance tasks:
1. Update dependencies
2. Check security vulnerabilities
3. Test functionality
4. Update documentation
5. Verify compatibility

## Support

For template development support:
1. Check existing templates
2. Review documentation
3. Test thoroughly
4. Submit detailed PRs 