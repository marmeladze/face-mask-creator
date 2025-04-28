# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] - 2024-04-29

### Added
- Non-interactive installation support for pip install
- Improved model download process
- Better error handling during installation

### Changed
- Updated installation instructions in README
- Modified setup.py to handle pip installs better
- Improved documentation clarity

### Fixed
- Fixed pip install failing due to interactive prompts
- Fixed model download process during installation

## [0.0.3] - 2024-03-21

### Added
- Interactive model download process with user choices
- Support for custom model paths during installation
- Smoke tests to verify installation
- Google Drive download support for BiSeNet model
- Configuration file to store model paths

### Changed
- Updated BiSeNet model download URL to Google Drive
- Improved error handling during model downloads
- Added gdown dependency for Google Drive downloads

### Fixed
- Fixed BiSeNet model download issues
- Improved model path handling and storage

## [0.0.2] - 2024-03-20

### Added
- Web interface extras package with Flask server
- Support for custom model paths
- Improved error handling and logging
- Configuration file for model paths

### Changed
- Improved installation process
- Enhanced documentation with web interface examples
- Better error messages for model loading issues

### Fixed
- Fixed model loading error handling
- Improved web interface error display
- Fixed import issues in web server

## [0.0.1] - 2024-03-19

### Added
- Initial release
- Basic face mask creation functionality
- Support for binary and colored masks
- Face region parsing with BiSeNet
- Landmark detection with dlib
- Simple command-line interface 