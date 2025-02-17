# Pentest Browser AI - Burp Suite Extension

A Burp Suite extension that provides AI-powered web penetration testing capabilities.

## Development Setup

1. Install Java Development Kit (JDK) 17 or later:
   - Download from: https://adoptium.net/
   - Add JAVA_HOME to environment variables
   - Add %JAVA_HOME%\bin to PATH

2. Install Apache Maven:
   - Download from: https://maven.apache.org/download.cgi
   - Extract to a directory (e.g., C:\Program Files\Apache\maven)
   - Add MAVEN_HOME environment variable pointing to the installation directory
   - Add %MAVEN_HOME%\bin to PATH

3. Build the extension:
   ```batch
   build.bat
   ```
   This will create the extension JAR file in the `target` directory.

## Loading in Burp Suite

1. Open Burp Suite Professional
2. Go to Extensions tab
3. Click "Add" button
4. Select "Java" as extension type
5. Click "Select file" and choose the generated JAR file:
   ```
   target/pentest-browser-extension-1.0-SNAPSHOT-jar-with-dependencies.jar
   ```
6. The extension should now appear in the extensions list
7. Enable "Use AI" checkbox in the extension settings

## Features

- AI-powered web penetration testing
- Automated vulnerability analysis
- Security testing assistance
- Request/response analysis

## Development

The extension is built using:
- Java 17
- Burp Suite Montoya API
- Maven for dependency management

## Project Structure

```
burp-extension/
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── pentest/
│                   └── browser/
│                       └── PentestBrowserExtension.java
├── pom.xml
├── build.bat
└── README.md
``` 