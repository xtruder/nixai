{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication overrides;
      in {
        packages = {
          nixai = mkPoetryApplication {
            projectDir = self;
            python = pkgs.python311;
            overrides = overrides.withDefaults
              (self: super: {
                pypika = super.pypika.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                  }
                );
                bcrypt = super.bcrypt.override {
                  preferWheel = false;
                };
                watchfiles = super.watchfiles.override {
                  preferWheel = false;
                };
                python-magic = super.python-magic.override {
                  preferWheel = false;
                };
                #bs4 = super.bs4.overridePythonAttrs
                #(
                #  old: {
                #    buildInputs = (old.buildInputs or [ ]) ++ [ super.hatchling ];
                #  }
                #);
                #tiktoken = super.tiktoken.overridePythonAttrs
                #(
                #  old: {
                #    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools-rust ];
                #    nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pkgs.rustc];
                #  }
                #);
                # Notice that using .overridePythonAttrs or .overrideAttrs wont work!
              });
            extras = [];
            checkGroups = [];
            preferWheels = true;
          };
          default = self.packages.${system}.nixai;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [ poetry python311 sqlite-interactive ];
        };
      });
}
