
{
  description = "Jupyter environment for STM line-by-line classification";
  
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      
      # Python with all required packages
      pythonEnv = pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
        # Jupyter
        jupyter
        ipykernel
        ipywidgets
        notebook
        
        # Deep Learning
        torch
        torchvision
        
        # Data Science
        numpy
        scipy
        pandas
        h5py
        
        # Visualization
        matplotlib
        seaborn
        plotly
        
        # Machine Learning Utils
        scikit-learn
        
        # Development
        python-lsp-server
        black
        isort
        flake8
        
        # Utilities
        tqdm
        
      ]);
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          buildInputs = [
            pythonEnv

            # Optional: CUDA support (uncomment if you have NVIDIA GPU)
            # pkgs.cudaPackages.cudatoolkit
            # pkgs.cudaPackages.cudnn

            # Utilities
            pkgs.git
          ];

          shellHook = ''
            echo "STM Classification Environment"
            echo "=================================="
            echo "Python: $(python --version)"
            echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
            echo ""
            echo "To start Jupyter:"
            echo "  jupyter notebook"
            echo ""
            echo "To start JupyterLab (recommended):"
            echo "  jupyter lab"
            echo ""

            # Set up Jupyter config directory in project
            export JUPYTER_CONFIG_DIR="./.jupyter"
            export JUPYTER_DATA_DIR="./.jupyter/data"

            # Create directories if they don't exist
            mkdir -p .jupyter/data

            # Ensure CUDA is available if installed
            # Uncomment and configure the line below if you need CUDA support
            # export LD_LIBRARY_PATH=<path-to-nvidia>:<path-to-cuda>:$LD_LIBRARY_PATH
          '';
        };

        # Alternative: JupyterLab-focused environment
        jupyterlab = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.nodejs  # For JupyterLab extensions
          ];

          shellHook = ''
            echo "STM Classification - JupyterLab Environment"
            echo "=============================================="
            jupyter lab --version
            echo ""
            echo "Starting JupyterLab..."
            jupyter lab --ip=127.0.0.1 --no-browser
          '';
        };

        # Minimal environment for quick experiments
        minimal = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
              jupyter
              jupytext
              notebook
              torch
              numpy
              matplotlib
              scikit-learn
              h5py
            ]))
          ];

          shellHook = ''
            echo "Minimal Jupyter environment loaded"
            echo "Run: jupyter notebook"
          '';
        };
      };
    };
}
