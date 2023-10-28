use proc_macro2::{Ident, Span};
use quote::{format_ident, quote};
use syn::{
    parse::Parse, parse_macro_input, spanned::Spanned, FnArg, ImplItem, ImplItemFn, ItemImpl, Meta,
    Pat, PatIdent, PatType, Path, Result, ReturnType, Type, TypePath,
};

#[proc_macro_attribute]
pub fn impl_gen(
    args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let _ = args;
    let gen_impl_data = parse_macro_input!(input as GenImplData);
    gen_impl_data
        .into_token_stream()
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
    // gen_parse(args.into(), input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

struct ProcessData {
    /// Name of the user writtin function to be called for process
    fn_name: Ident,
    inputs: Vec<Ident>,
    outputs: Vec<Ident>,
    parameters: Vec<Parameter>,
}

struct NewData {
    fn_name: Ident,
    parameters: Vec<PatType>,
}
impl NewData {
    fn into_tokens(self, type_ident: &Ident, handle_name: &Ident) -> proc_macro2::TokenStream {
        let NewData {
            fn_name,
            parameters,
        } = self;

        let create_fn_name = format_ident!("{}", type_ident.to_string().to_lowercase());
        let param_types_in_sig = parameters.iter().map(|p| quote! {#p});
        let param_names_in_call = parameters.iter().map(|p| {
            let ident = *p.pat.clone();
            quote! {#ident}
        });
        quote! {
                    // Init handle fn
                    // TODO: Take parameters to new() to be able to call it
                    pub fn #create_fn_name(#(#param_types_in_sig),*) -> knyst::handles::NodeHandle<#handle_name> {
                        use knyst::controller::KnystCommands;
                        let node_id =
                            knyst::modal_interface::commands().push_without_inputs(#type_ident::#fn_name(#(#param_names_in_call),*));
                        knyst::handles::NodeHandle::new(#handle_name{node_id})
                    }
        }
    }
}
struct InitData {
    fn_name: Ident,
    parameters: Vec<Parameter>,
}
impl InitData {
    fn into_tokens(self) -> proc_macro2::TokenStream {
        let InitData {
            fn_name,
            parameters,
        } = self;

        let parameters_assignments = parameters.iter().map(|p| {
            let p_ident = &p.ident;
            match p._ty {
                ParameterTy::Input => todo!(),
                ParameterTy::Output => todo!(),
                ParameterTy::SampleRate => {
                    quote! { let #p_ident: knyst::prelude::SampleRate = sample_rate.into();}
                }
                ParameterTy::ResourcesShared => quote! { #p_ident = resources;},
                ParameterTy::ResourcesMut => quote! { #p_ident = resources;},
                ParameterTy::BlockSize => quote! {
                    let #p_ident: knyst::prelude::BlockSize = block_size.into();
                },
            }
        });
        let parameters_in_sig = parameters.iter().map(|p| &p.ident);
        quote! {
            fn init(&mut self, block_size: usize, sample_rate: Sample) {
                #(#parameters_assignments)*
                self.#fn_name(#(#parameters_in_sig),*);
            }

        }
    }
}

struct GenImplData {
    /// The last segment of the type_path, which should be used for the function shorthand
    type_ident: Ident,
    /// Used to refer to the same full path for the type as in the original impl block
    type_path: Path,
    process_data: ProcessData,
    init_data: Option<InitData>,
    /// Parameters to the new function
    new_data: Option<NewData>,
    org_item_impl: ItemImpl,
}

impl GenImplData {
    fn into_token_stream(self) -> Result<proc_macro2::TokenStream> {
        let GenImplData {
            type_ident,
            type_path,
            process_data,
            init_data,
            org_item_impl,
            new_data,
        } = self;
        let ProcessData {
            fn_name: process_fn_name,
            inputs,
            outputs,
            parameters,
        } = process_data;
        let init_function = if let Some(init_data) = init_data {
            init_data.into_tokens()
        } else {
            quote! {
                fn init(&mut self, _block_size: usize, _sample_rate: Sample) {}
            }
        };
        let parameters_in_sig = parameters
            .iter()
            .map(|p| Ident::new(&format!("__impl_gen_{}", p.ident), Span::call_site()));
        let num_inputs = inputs.len();
        let num_outputs = outputs.len();
        let type_name_string = type_ident.to_string();
        let match_input_names = inputs.iter().enumerate().map(|(i, name)| {
            let name_string = name.to_string();
            quote! { #i => #name_string, }
        });
        let match_output_names = outputs.iter().enumerate().map(|(i, name)| {
            let name_string = name.to_string();
            quote! { #i => #name_string, }
        });
        let extract_inputs = inputs.iter().enumerate().map(|(i, ident)| {
            let ident = Ident::new(&format!("__impl_gen_{ident}"), Span::call_site());
            quote! { let #ident = inputs.get_channel(#i); }
        });
        let extract_outputs = outputs.iter().map(|i| {
            let i = Ident::new(&format!("__impl_gen_{i}"), Span::call_site());
            quote! { let #i = outputs.next().unwrap(); }
        });
        let extract_other_process_parameters = parameters.iter().map(|p| {
            let p_ident = &Ident::new(&format!("__impl_gen_{}", p.ident), Span::call_site());
            match p._ty {
                ParameterTy::Input | ParameterTy::Output => quote! {},
                ParameterTy::SampleRate => {
                    quote! { let #p_ident: knyst::prelude::SampleRate = ctx.sample_rate.into(); }
                }
                ParameterTy::ResourcesShared => quote! {
                     let #p_ident: &knyst::prelude::Resources = resources;
                },
                ParameterTy::ResourcesMut => quote! {
                     let #p_ident: &mut knyst::prelude::Resources = resources;
                },
                ParameterTy::BlockSize => quote! {
                    let #p_ident: knyst::prelude::BlockSize = ctx.block_size().into();
                },
            }
        });

        // let handle_name = format_ident!("{type_ident}Handle");
        let handle_name = Ident::new(&format!("{}Handle", type_ident), Span::call_site());
        let init_handle_fn = new_data.map(|nd| nd.into_tokens(&type_ident, &handle_name));
        let handle_functions = inputs.iter().map(|i| {
            let param_ident = Ident::new(&i.to_string().to_lowercase(), Span::call_site());
            let param_string = i.to_string();
            quote! {
                        pub fn #i(&self, #param_ident: impl Into<knyst::handles::Input>) -> knyst::handles::NodeHandle<Self> {
                            use knyst::controller::KnystCommands;
            let inp = #param_ident.into();
            match inp {
                knyst::handles::Input::Constant(v) => {
                            knyst::modal_interface::commands().connect(knyst::graph::connection::constant(v).to(self.node_id).to_channel(#param_string));
                }
                knyst::handles::Input::Handle { output_channels } => {
                    for (i, (node_id, chan)) in output_channels.enumerate() {
                            knyst::modal_interface::commands().connect(node_id.to(self.node_id).from_channel(chan).to_channel(#param_string));
                    }
                }
            }
            knyst::handles::NodeHandle::new(*self)
                        }
                    }
        });
        Ok(quote! {
                    #org_item_impl

                    impl knyst::prelude::Gen for #type_path {
                        fn process(&mut self, ctx: knyst::prelude::GenContext, resources: &mut knyst::prelude::Resources) -> knyst::prelude::GenState {
                            #(#extract_other_process_parameters)*
                            let mut inputs = ctx.inputs;
                            #(#extract_inputs)*

                            let mut outputs = ctx.outputs.iter_mut();
                            #(#extract_outputs)*

                            self.#process_fn_name ( #(#parameters_in_sig),* )
                        }

            fn num_inputs(&self) -> usize {
                #num_inputs
            }
            fn num_outputs(&self) -> usize {
                #num_outputs
            }
            fn input_desc(&self, input: usize) -> &'static str {
                match input {
                    #(#match_input_names)*
                    _ => ""
                }
            }
            fn output_desc(&self, output: usize) -> &'static str {
                match output {
                    #(#match_output_names)*
                    _ => ""
                }
            }
            #init_function
            fn name(&self) -> &'static str {
                #type_name_string
            }
                    }

                    // Handle
                    #[derive(Copy, Clone, Debug)]
                    pub struct #handle_name {
                        node_id: knyst::prelude::NodeId,
                    }
                    impl #handle_name {
                        #(#handle_functions)*
                    }
                    impl knyst::handles::NodeHandleData for #handle_name {
        fn out_channels(&self) -> knyst::handles::ChannelIter {
            knyst::handles::ChannelIter::single_node_id(
                self.node_id,
                #num_outputs,
            )
        }

        fn in_channels(&self) -> knyst::handles::ChannelIter {
            knyst::handles::ChannelIter::single_node_id(
                self.node_id,
                #num_inputs,
            )
        }

        fn node_ids(&self) -> knyst::handles::NodeIdIter {
            knyst::handles::NodeIdIter::Single(self.node_id)
        }

                    }

                    #init_handle_fn
                })
    }
}

impl Parse for GenImplData {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let mut item_impl: ItemImpl = input.parse()?;
        let ty = *item_impl.self_ty.clone();
        let Type::Path(TypePath {
            path: type_path, ..
        }) = ty
        else {
            return Err(syn::Error::new(
                ty.span(),
                "this type of impl is not supported",
            ));
        };
        let type_ident = {
            type_path
                .segments
                .first()
                .ok_or(syn::Error::new(type_path.span(), "No segment in path"))?
                .clone()
                .ident
        };

        let mut process_data = None;
        let mut init_data = None;
        let mut new_data = None;

        let full_item_span = item_impl.span();

        for item in &mut item_impl.items {
            if let ImplItem::Fn(ref mut impl_item_fn) = item {
                let mut remove_attributes = vec![];
                // Does this function have an attribute we recognise?
                for (attr_i, attr) in impl_item_fn.attrs.iter().enumerate() {
                    if let Meta::Path(p) = &attr.meta {
                        if let Some(path_segment) = p.segments.first() {
                            match path_segment.ident.to_string().as_ref() {
                                "process" => {
                                    remove_attributes.push(attr_i);
                                    process_data = Some(parse_process_fn(impl_item_fn)?);
                                }
                                "init" => {
                                    remove_attributes.push(attr_i);
                                    init_data = Some(parse_init_fn(impl_item_fn)?);
                                }
                                "new" => {
                                    remove_attributes.push(attr_i);
                                    new_data = Some(parse_new_fn(impl_item_fn)?);
                                }
                                _ => (),
                            }
                        }
                    }
                }
                for i in remove_attributes.iter().rev() {
                    impl_item_fn.attrs.remove(*i);
                }
            }
        }

        let Some(process_data) = process_data else {
            return Err(syn::Error::new(
                full_item_span,
                "No #[process] method in the block",
            ));
        };

        // let ItemImpl::Type(ItemImpl { ident: type_ident, ty, .. }) = impl_item else {
        //     return Err(syn::Error::new(impl_item.span(), "Invalid impl block"));
        // };
        Ok(GenImplData {
            type_path,
            org_item_impl: item_impl,
            type_ident,
            process_data,
            init_data,
            new_data,
        })
    }
}

// - `&[Sample]` : input
// - `&mut [Sample]` : output
// - `&Resources` : immutable access to Resources
// - `&mut Resources`: mutable access to Resources
// - `BlockSize`
// - `SampleRate`
// - `&mut MessageSender` : Message output. Direct function call message sending to Gens this is connected to, or adding them to a channel for buffering if messages are sent to a Graph output
enum ParameterTy {
    Input,
    Output,
    SampleRate,
    ResourcesShared,
    ResourcesMut,
    BlockSize,
}

struct Parameter {
    _ty: ParameterTy,
    ident: Ident,
}

fn parse_process_fn(impl_item_fn: &ImplItemFn) -> Result<ProcessData> {
    let mut inputs = vec![];
    let mut outputs = vec![];
    let mut parameters = vec![];

    let ReturnType::Type(_, return_type) = &impl_item_fn.sig.output else {
        return Err(syn::Error::new(
            impl_item_fn.sig.output.span(),
            "#[process] method needs to return a GenState",
        ));
    };
    let Type::Path(TypePath {
        path: Path { segments, .. },
        ..
    }) = &**return_type
    else {
        return Err(syn::Error::new(
            return_type.span(),
            "#[process] method needs to return a GenState",
        ));
    };
    if segments.last().unwrap().ident != "GenState" {
        return Err(syn::Error::new(
            return_type.span(),
            "#[process] method needs to return a GenState",
        ));
    }
    let process_fn_name = impl_item_fn.sig.ident.clone();
    for arg in &impl_item_fn.sig.inputs {
        if let FnArg::Typed(param) = arg {
            let Pat::Ident(PatIdent { ident: name, .. }) = &*param.pat else {
                return Err(syn::Error::new(param.span(), "Unsupported param"));
            };
            let parameter = parse_parameter(param, name)?;
            match parameter._ty {
                ParameterTy::Input => inputs.push(parameter.ident.clone()),
                ParameterTy::Output => outputs.push(parameter.ident.clone()),
                _ => (),
            }
            parameters.push(parameter);
        }
    }
    Ok(ProcessData {
        fn_name: process_fn_name,
        inputs,
        outputs,
        parameters,
    })
}

fn parse_init_fn(impl_item_fn: &ImplItemFn) -> Result<InitData> {
    let mut inputs = vec![];
    let mut outputs = vec![];
    let mut parameters = vec![];

    if let ReturnType::Default = impl_item_fn.sig.output {
    } else {
        return Err(syn::Error::new(
            impl_item_fn.sig.output.span(),
            "#[init] method should return nothing",
        ));
    }
    let fn_name = impl_item_fn.sig.ident.clone();
    for arg in &impl_item_fn.sig.inputs {
        if let FnArg::Typed(param) = arg {
            let Pat::Ident(PatIdent { ident: name, .. }) = &*param.pat else {
                return Err(syn::Error::new(param.span(), "Unsupported param"));
            };
            let parameter = parse_parameter(param, name)?;
            match parameter._ty {
                ParameterTy::Input => inputs.push(parameter.ident.clone()),
                ParameterTy::Output => outputs.push(parameter.ident.clone()),
                _ => (),
            }
            parameters.push(parameter);
        }
    }
    Ok(InitData {
        parameters,
        fn_name,
    })
}

fn parse_new_fn(impl_item_fn: &ImplItemFn) -> Result<NewData> {
    let mut parameters = vec![];

    // TODO: Check that the function returns Self
    // if let ReturnType::Default = impl_item_fn.sig.output {
    // } else {
    //     return Err(syn::Error::new(
    //         impl_item_fn.sig.output.span(),
    //         "#[init] method should return nothing",
    //     ));
    // }
    let fn_name = impl_item_fn.sig.ident.clone();
    for arg in &impl_item_fn.sig.inputs {
        if let FnArg::Typed(param) = arg {
            parameters.push(param.clone());
        }
    }
    Ok(NewData {
        parameters,
        fn_name,
    })
}

fn parse_parameter(param: &PatType, name: &Ident) -> Result<Parameter> {
    match *param.ty {
        Type::Reference(ref ty) => {
            match &*(ty.elem) {
                Type::Slice(ref slice_type) => {
                    match *slice_type.elem {
                        Type::Path(ref p) if p.path.segments.first().unwrap().ident == "Sample" => {
                        }
                        _ => {
                            return Err(syn::Error::new(slice_type.elem.span(), "Unknown input"));
                        }
                    }
                    // The type is okay to be an input or output
                    if ty.mutability.is_some() {
                        // outputs.push(name.clone());
                        Ok(Parameter {
                            _ty: ParameterTy::Output,
                            ident: name.clone(),
                        })
                    } else {
                        // inputs.push(name.clone());
                        Ok(Parameter {
                            _ty: ParameterTy::Input,
                            ident: name.clone(),
                        })
                    }
                }
                Type::Path(ty_path) => {
                    let ty_ident = ty_path
                        .path
                        .segments
                        .last()
                        .map(|seg| seg.ident.to_string());
                    match ty_ident.as_deref() {
                        Some("SampleRate") => Ok(Parameter {
                            ident: name.clone(),
                            _ty: ParameterTy::SampleRate,
                        }),
                        Some("Resources") => Ok(Parameter {
                            ident: name.clone(),
                            _ty: if ty.mutability.is_some() {
                                ParameterTy::ResourcesMut
                            } else {
                                ParameterTy::ResourcesShared
                            },
                        }),
                        _ => Err(syn::Error::new(
                            ty.span(),
                            "Unsupported type in knyst method.",
                        )),
                    }
                }
                _ => Err(syn::Error::new(
                    ty.span(),
                    "Unsupported type in knyst method.",
                )),
            }
        }

        Type::Path(ref ty) => {
            match ty
                .path
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
                .as_deref()
            {
                Some("SampleRate") => Ok(Parameter {
                    ident: name.clone(),
                    _ty: ParameterTy::SampleRate,
                }),
                Some("BlockSize") => Ok(Parameter {
                    ident: name.clone(),
                    _ty: ParameterTy::BlockSize,
                }),
                _ => Err(syn::Error::new(
                    param.ty.span(),
                    "Unsupported type in knyst method.",
                )),
            }
        } // TODO: Other types
        _ => Err(syn::Error::new(
            param.ty.span(),
            "Unsupported type in knyst method.",
        )),
    }
}
