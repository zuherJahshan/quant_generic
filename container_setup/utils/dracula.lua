return {
  {
    "Mofiqul/dracula.nvim",
    priority = 1000, -- Load before other plugins
    config = function()
      require("dracula").setup({
        -- Optional: custom settings
        show_end_of_buffer = true, -- show ~ after the end of buffers
        transparent_bg = false,
        italic_comment = true,
      })
      vim.cmd.colorscheme("dracula")
    end,
  },
}
