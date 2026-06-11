with open('src/bridge/memory.rs', 'r') as f:
    code = f.read()

code = code.replace("use crate::vfs::VirtualFileSystem;", "#[cfg(not(test))] use crate::vfs::VirtualFileSystem;\n#[cfg(test)] use agentic_kernel::vfs::VirtualFileSystem;")

with open('src/bridge/memory.rs', 'w') as f:
    f.write(code)
