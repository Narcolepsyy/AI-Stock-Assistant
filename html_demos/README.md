# HTML Demos

This directory contains HTML demonstration files for showcasing UI/UX features of the Azure-OpenAI Stock Analysis Tool.

## Available Demos

Currently, this directory is prepared for HTML demo files but none exist yet in the codebase.

Expected demo types include:
- Error handling UI demonstrations
- Interactive component showcases
- QA sticky header implementations
- UI/UX prototypes and mockups

## Usage

HTML demos can be opened directly in a browser:

```bash
# Open with default browser
xdg-open html_demos/demo_file.html

# Or with specific browser
firefox html_demos/demo_file.html
chrome html_demos/demo_file.html
```

## Serving HTML Demos

For demos requiring a web server (e.g., for CORS, WebSocket testing):

```bash
# Using Python's built-in server
cd html_demos
python -m http.server 8080

# Then open http://localhost:8080/demo_file.html
```

## Purpose

HTML demos serve to:
- Showcase UI/UX features independently of the main application
- Prototype new interface components
- Demonstrate error handling patterns
- Test responsive design across devices
- Provide interactive examples for documentation

## Structure

HTML demos should be self-contained:
- Inline CSS or linked stylesheets
- Inline JavaScript or linked scripts
- No external dependencies when possible
- Clear documentation in HTML comments

## Contributing

When adding new HTML demos:
1. Use descriptive filenames (e.g., `error_handling_demo.html`, `chart_visualization_demo.html`)
2. Include metadata in HTML `<head>`:
   ```html
   <meta name="description" content="Demo description">
   <meta name="author" content="Your Name">
   ```
3. Add comments explaining key sections
4. Test in multiple browsers (Chrome, Firefox, Safari)
5. Ensure responsive design for mobile devices
6. Update this README with the new demo details
7. Consider adding screenshots to `docs/` for reference

## Related Resources

- **Documentation**: See `../docs/` for detailed feature documentation
- **React Components**: See `../frontend/src/components/` for production UI components
- **Static Assets**: See `../static/` for shared CSS/JS/images
