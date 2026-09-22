# Neovim remote machine installer
cd ~
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz
tar -xzf nvim-linux-x86_64.tar.gz
cp -r nvim-linux-x86_64/* ~/.local/
rm -rf nvim-linux-x86_64*
export PATH="$HOME/.local/bin:$PATH"

# Setup lazyvim
git clone https://github.com/LazyVim/starter ~/.config/nvim
