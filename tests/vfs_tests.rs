use agentic_kernel::vfs::VirtualFileSystem;

#[test]
fn test_virtual_file_system_operations() {
    let mut vfs = VirtualFileSystem::new(64 * 1024 * 1024);

    // 1. Write and read file
    vfs.write_file("data/config.json", "{\"setting\": true}");
    let content = vfs
        .read_file("data/config.json")
        .expect("File should exist");
    assert_eq!(content, "{\"setting\": true}");

    // 2. Edit file (append)
    vfs.edit_file("data/log.txt", "line 1\n");
    vfs.edit_file("data/log.txt", "line 2\n");
    let log_content = vfs.read_file("data/log.txt").unwrap();
    assert_eq!(log_content, "line 1\nline 2\n");

    // 3. List directory
    vfs.write_file("data/models/agent.bin", "binary_data");
    let dir_list = vfs.list_directory("data/");
    assert_eq!(dir_list.len(), 3);
    assert!(dir_list.contains(&"data/config.json".to_string()));
    assert!(dir_list.contains(&"data/log.txt".to_string()));
    assert!(dir_list.contains(&"data/models/agent.bin".to_string()));

    // 4. Regex search
    let search_results = vfs.search_files("data/.*\\.txt");
    assert_eq!(search_results.len(), 1);
    assert_eq!(search_results[0], "data/log.txt");
}
