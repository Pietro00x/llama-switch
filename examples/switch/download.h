#pragma once

#include <cstddef>
#include <functional>
#include <string>

// Rewrites huggingface.co ".../blob/main/..." page links to ".../resolve/main/..." download links.
std::string normalize_hf_url(std::string url);

// Called while the body is received. total is 0 when the server did not send Content-Length.
// Return false to cancel the download (download_file then fails with "download cancelled").
using download_progress_cb = std::function<bool(size_t received, size_t total)>;

// Downloads url (http or https, follows up to 6 redirects) to destination.
// Creates parent directories; removes the partial file on failure.
bool download_file(
    const std::string & url,
    const std::string & destination,
    std::string & err,
    const download_progress_cb & progress_cb);

// Closes the socket/ssl services if download_file initialized them.
void download_exit();
