git status --porcelain | ForEach-Object {
    if ($_ -notmatch "__pycache__" -and $_ -notmatch "^D") {
        $parts = $_ -split "\s+"
        $file = $parts[-1]
        git add $file
    }
}