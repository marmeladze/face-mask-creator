# Changelog

All notable changes to the face-mask-creator package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2024-03-21

### Added
- Web interface extras package with Flask-based server
- Interactive web UI for mask creation
- Support for custom model paths during installation
- Option to skip model downloads during installation
- Configuration file to store model paths

### Changed
- Updated BiSeNet model URL to point to the official repository
- Improved installation process with better error handling
- Enhanced documentation with web interface usage instructions

### Fixed
- Fixed broken BiSeNet model download URL
- Improved error handling during model downloads
- Fixed pip installation issues with command-line arguments

## [0.0.1] - 2024-03-20

### Added
- Initial release
- Basic face mask creation functionality
- Support for dlib's facial landmarks detection
- Support for BiSeNet face parsing
- Automatic model downloads during installation 