with open('src/bridge/memory.rs', 'r') as f:
    code = f.read()

code = code.replace("#[cfg(test)] use agentic_kernel::vfs::VirtualFileSystem;", "use crate::vfs::VirtualFileSystem;")
code = code.replace("#[cfg(not(test))] use crate::vfs::VirtualFileSystem;", "")

with open('src/bridge/memory.rs', 'w') as f:
    f.write(code)
