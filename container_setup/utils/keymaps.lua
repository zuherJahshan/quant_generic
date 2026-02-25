-- Keymaps are automatically loaded on the VeryLazy event
-- Default keymaps that are always set: https://github.com/LazyVim/LazyVim/blob/main/lua/lazyvim/config/keymaps.lua
-- Add any additional keymaps here

-- map <C-w> to copilot word competion and <C-Space> to copilot completion
vim.g.copilot_no_tab_map = true
vim.keymap.set("i", "<C-w>", "<Plug>(copilot-accept-word)")
vim.keymap.set("i", "<C-Space>", 'copilot#Accept("\\<CR>")', {
  expr = true,
  replace_keycodes = false,
})

