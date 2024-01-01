use proc_macro2::{Ident, Span};
use quote::{format_ident, quote};
use syn::{
    parse::Parse, parse_macro_input, punctuated::Punctuated, spanned::Spanned, ExprAssign,
    ExprPath, FnArg, ImplItem, ImplItemFn, ItemImpl, Meta, Pat, PatIdent, PatType, Path, Result,
    ReturnType, Token, Type, TypePath,
};

#[proc_macro_attribute]
pub fn impl_gen(
    args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let arg_data = parse_macro_input!(args as ArgData);
    let gen_impl_data = parse_macro_input!(input as GenImplData);
    gen_impl_data
        .into_token_stream(arg_data)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
    // gen_parse(args.into(), input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

struct ProcessData {
    /// Name of the user written function to be called for process
    fn_name: Ident,
    inputs: Vec<Ident>,
    outputs: Vec<Ident>,
    parameters: Vec<Parameter>,
}

struct ArgData {
    range: Option<Range>,
}
impl Parse for ArgData {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let mut range = None;
        let assigns = Punctuated::<ExprAssign, Token![,]>::parse_terminated(input)?;
        for assign in assigns {
            match &*assign.left {
                syn::Expr::Path(ExprPath {
                    attrs: _,
                    qself: _,
                    path,
                }) => {
                    let name = path.segments.first().map(|s| s.ident.to_string());
                    match name.as_deref() {
                        Some("range") => {
                            let value = get_expr_path_ident(&assign.right)?;
                            match value.to_string().as_str() {
                                "normal" => range = Some(Range::Normal),
                                "positive" => range = Some(Range::Positive),
                                _ => {
                                    return Err(syn::Error::new(
                                        assign.right.span(),
                                        "unsupported value to range",
                                    ))
                                }
                            }
                        }
                        _ => {
                            return Err(syn::Error::new(
                                assign.left.span(),
                                "unsupported argument to macro",
                            ))
                        }
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        assign.left.span(),
                        "unsupported argument to macro",
                    ))
                }
            }
        }
        Ok(ArgData { range })
    }
}
fn get_expr_path_ident(expr: &syn::Expr) -> Result<&Ident> {
    match expr {
        syn::Expr::Path(ExprPath {
            attrs: _,
            qself: _,
            path,
        }) => path
            .segments
            .first()
            .map(|s| Ok(&s.ident))
            .unwrap_or_else(|| Err(syn::Error::new(expr.span(), "no segment in path"))),
        _ => Err(syn::Error::new(expr.span(), "unexpected value")),
    }
}

#[derive(Clone, Copy)]
enum Range {
    /// [-1,1]
    Normal,
    /// [0,1]
    Positive,
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

        let create_fn_name = format_ident!(
            "{}",
            ident_case::RenameRule::SnakeCase.apply_to_variant(type_ident.to_string())
        );
        let param_types_in_sig = parameters.iter().map(|p| quote! {#p});
        let param_names_in_call = parameters.iter().map(|p| {
            let ident = *p.pat.clone();
            quote! {#ident}
        });
        let doc_str = format!("Upload a {type_ident} and return a handle to it.");
        quote! {
            // Init handle fn
            #[doc = #doc_str]
            pub fn #create_fn_name(#(#param_types_in_sig),*) -> knyst::handles::Handle<#handle_name> {
                use knyst::controller::KnystCommands;
                let node_id =
                    knyst::modal_interface::knyst_commands().push_without_inputs(#type_ident::#fn_name(#(#param_names_in_call),*));
                knyst::handles::Handle::new(#handle_name{node_id})
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
                ParameterTy::InputTrig => todo!(),
                ParameterTy::OutputTrig => todo!(),
                ParameterTy::NodeId => {
                    quote! { let #p_ident = node_id; }
                }
            }
        });
        let parameters_in_sig = parameters.iter().map(|p| &p.ident);
        quote! {
            fn init(&mut self, block_size: usize, sample_rate: knyst::Sample, node_id: knyst::prelude::NodeId) {
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
    fn into_token_stream(self, arg_data: ArgData) -> Result<proc_macro2::TokenStream> {
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
        let ArgData { range } = arg_data;
        let init_function = if let Some(init_data) = init_data {
            init_data.into_tokens()
        } else {
            quote! {
                fn init(&mut self, _block_size: usize, _sample_rate: knyst::Sample, _node_id: knyst::prelude::NodeId) {}
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
                ParameterTy::Input
                | ParameterTy::Output
                | ParameterTy::InputTrig
                | ParameterTy::OutputTrig
                | ParameterTy::NodeId => quote! {},
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
        let handle_range_impl = range.map(|range| match range {
            Range::Normal => {
                quote! {
                    impl knyst::handles::HandleNormalRange for #handle_name {}
                }
            }
            Range::Positive => todo!(),
        });
        let handle_functions = parameters.iter().filter(|&p| matches!(p._ty, ParameterTy::Input | ParameterTy::InputTrig)).map(|p| {
            let param_ident = Ident::new(&p.ident.to_string().to_lowercase(), Span::call_site());
            let param_string = p.ident.to_string();
            let i = &p.ident;
            let doc_str = format!("Set the input {i} to the output of a handle or a constant value.");
            let set_fn = quote! {
                #[doc = #doc_str]
                pub fn #i(self, #param_ident: impl Into<knyst::handles::Input>) -> knyst::handles::Handle<Self> {
                    use knyst::controller::KnystCommands;
                    let inp = #param_ident.into();
                    match inp {
                        knyst::handles::Input::Constant(v) => {
                            let change = knyst::graph::ParameterChange {
                                time: knyst::graph::Time::Immediately,
                                input: (self.node_id,
                                    knyst::graph::connection::NodeChannel::Label(#param_string)).into(),
                                value: knyst::graph::Change::Constant(v),
                            };
                            knyst::modal_interface::knyst_commands().schedule_change(change);
                            // knyst::modal_interface::knyst_commands().connect(knyst::graph::connection::constant(v).to(self.node_id).to_channel(#param_string));
                        }
                        knyst::handles::Input::Handle { output_channels } => {
                            for (i, (node_id, chan)) in output_channels.enumerate() {
                            knyst::modal_interface::knyst_commands().connect(node_id.to(self.node_id).from_channel(chan).to_channel(#param_string));
                            }
                        }
                    }
                    knyst::handles::Handle::new(self)
                }
            };
            match p._ty {
                ParameterTy::Input => quote!{#set_fn},
                ParameterTy::InputTrig => {
                    let trig_fn_name = format_ident!("{}_trig", i);
                    let doc_str = format!("Send a trigger to the {i} input.");
                    // TODO: Move SimultaneousChanges into a modal setting on the commands.
                    quote!{
                        #set_fn
                        #[doc = #doc_str]
                        pub fn #trig_fn_name(self) -> knyst::handles::Handle<Self> {
                            use knyst::handles::HandleData;
                            for id in self.node_ids() {
                                let mut s = knyst::graph::SimultaneousChanges::now();
                                s.push(id.change().trigger(#param_string));
                                knyst::knyst_commands().schedule_changes(s);
                            }
                            knyst::handles::Handle::new(self)
                        }
                    }
                },
                _ => unreachable!()
            }
        });

        let handle_struct_doc_str = format!("Handle to a {type_ident}.");
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

                            impl #type_path {
                                /// Upload the Gen to the currently selected Graph and return a handle
                                pub fn upload(self) -> knyst::handles::Handle< #handle_name > {
                use knyst::controller::KnystCommands;
                let node_id =
                    knyst::modal_interface::knyst_commands().push_without_inputs(self);
                knyst::handles::Handle::new(#handle_name{node_id})
                                }
                            }

                            // Handle
                            #[doc = #handle_struct_doc_str]
                            #[derive(Copy, Clone, Debug)]
                            pub struct #handle_name {
                                node_id: knyst::prelude::NodeId,
                            }
                            impl #handle_name {
                                #(#handle_functions)*
                            }
                            impl knyst::handles::HandleData for #handle_name {
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

                            impl Into<knyst::handles::GenericHandle> for #handle_name {
            fn into(self) -> knyst::handles::GenericHandle {
                knyst::handles::GenericHandle::new(
                    self.node_id,
                    #num_inputs,
                    #num_outputs,
                )
        }
                            }

                            #init_handle_fn

                            #handle_range_impl
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
                let mut handled_through_attribute = false;
                for (attr_i, attr) in impl_item_fn.attrs.iter().enumerate() {
                    if let Meta::Path(p) = &attr.meta {
                        if let Some(path_segment) = p.segments.first() {
                            match path_segment.ident.to_string().as_ref() {
                                "process" => {
                                    if process_data.is_some() {
                                        return Err(syn::Error::new(
                                            impl_item_fn.span(),
                                            "Multiple process functions in impl.",
                                        ));
                                    }
                                    remove_attributes.push(attr_i);
                                    process_data = Some(parse_process_fn(impl_item_fn)?);
                                    handled_through_attribute = true;
                                }
                                "init" => {
                                    if init_data.is_some() {
                                        return Err(syn::Error::new(
                                            impl_item_fn.span(),
                                            "Multiple init functions in impl.",
                                        ));
                                    }
                                    remove_attributes.push(attr_i);
                                    init_data = Some(parse_init_fn(impl_item_fn)?);
                                    handled_through_attribute = true;
                                }
                                "new" => {
                                    if new_data.is_some() {
                                        return Err(syn::Error::new(
                                            impl_item_fn.span(),
                                            "Multiple new functions in impl.",
                                        ));
                                    }
                                    remove_attributes.push(attr_i);
                                    new_data = Some(parse_new_fn(impl_item_fn)?);
                                    handled_through_attribute = true;
                                }
                                _ => (),
                            }
                        }
                    }
                }
                for i in remove_attributes.iter().rev() {
                    impl_item_fn.attrs.remove(*i);
                }
                if !handled_through_attribute {
                    // Fn didn't have a recognised attribute. Check if the function name matches an attribute name instead.
                    let fn_name = &impl_item_fn.sig.ident;
                    match fn_name.to_string().as_str() {
                        "process" => {
                            if process_data.is_some() {
                                return Err(syn::Error::new(
                                    impl_item_fn.span(),
                                    "Multiple process functions in impl.",
                                ));
                            }
                            process_data = Some(parse_process_fn(impl_item_fn)?);
                        }
                        "init" => {
                            if init_data.is_some() {
                                return Err(syn::Error::new(
                                    impl_item_fn.span(),
                                    "Multiple init functions in impl.",
                                ));
                            }
                            init_data = Some(parse_init_fn(impl_item_fn)?);
                        }
                        "new" => {
                            if new_data.is_some() {
                                return Err(syn::Error::new(
                                    impl_item_fn.span(),
                                    "Multiple new functions in impl.",
                                ));
                            }
                            new_data = Some(parse_new_fn(impl_item_fn)?);
                        }
                        _ => (),
                    }
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
    /// Same as Input, but will create a special set function on the handle
    InputTrig,
    OutputTrig,
    /// The NodeId of the node, only available in init
    NodeId,
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
                ParameterTy::Input | ParameterTy::InputTrig => inputs.push(parameter.ident.clone()),
                ParameterTy::Output | ParameterTy::OutputTrig => {
                    outputs.push(parameter.ident.clone())
                }
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
                        Type::Path(ref p) if p.path.segments.first().unwrap().ident == "Trig" => {
                            // The type is okay to be an input or output
                            if ty.mutability.is_some() {
                                // outputs.push(name.clone());
                                Ok(Parameter {
                                    _ty: ParameterTy::OutputTrig,
                                    ident: name.clone(),
                                })
                            } else {
                                // inputs.push(name.clone());
                                Ok(Parameter {
                                    _ty: ParameterTy::InputTrig,
                                    ident: name.clone(),
                                })
                            }
                        }
                        _ => {
                            return Err(syn::Error::new(slice_type.elem.span(), "Unknown input"));
                        }
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
                        Some("NodeId") => Ok(Parameter {
                            ident: name.clone(),
                            _ty: ParameterTy::NodeId,
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
                Some("NodeId") => Ok(Parameter {
                    ident: name.clone(),
                    _ty: ParameterTy::NodeId,
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
